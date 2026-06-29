"""
COMPX523-26A Assignment 2: Quick Test with Sample Data

Quick validation of classifiers against the instructor's sample CSV.
Focuses on the tournament metrics: Balanced Accuracy + Macro F1.
"""

import os
import sys
import time
import numpy as np

# Set JAVA_HOME before importing capymoa — portable across machines.
_DEFAULT_JAVA_HOMES = [
    "/opt/homebrew/Cellar/openjdk@17/17.0.18/libexec/openjdk.jdk/Contents/Home",
    "/usr/lib/jvm/java-17-openjdk-arm64",
    "/usr/lib/jvm/java-11-openjdk-arm64",
    "/usr/lib/jvm/default-java",
]
if not os.environ.get("JAVA_HOME") or not os.path.exists(os.environ["JAVA_HOME"]):
    for _candidate in _DEFAULT_JAVA_HOMES:
        if os.path.exists(_candidate):
            os.environ["JAVA_HOME"] = _candidate
            break

from capymoa.stream import CSVStream
from capymoa.classifier import NaiveBayes

# Import our classifiers
from ensemble_classifier import EnsembleClassifier
from champion_classifier import ChampionClassifier


def create_csv_stream(csv_path):
    """Create a stream from the sample CSV file."""
    return CSVStream(
        file=csv_path,
        target="class",
        categories={"class": ["0", "1"]},
        name="sample_test",
        length=37440,
    )


def compute_metrics(y_true, y_pred):
    """Compute all tournament-relevant metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    accuracy = np.mean(y_true == y_pred)

    recalls = []
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c))

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        recalls.append(rec)
        f1s.append(f1)

    balanced_acc = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    tournament = 0.5 * balanced_acc + 0.5 * macro_f1

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'tournament_score': tournament,
        'per_class_recall': dict(zip(classes.tolist(), recalls)),
    }


def test_classifier(csv_path, schema, make_classifier, name, max_inst=None):
    """Run prequential evaluation and return metrics."""
    stream = create_csv_stream(csv_path)
    clf = make_classifier(schema)

    y_true, y_pred = [], []
    count = 0
    t0 = time.time()

    while stream.has_more_instances():
        if max_inst and count >= max_inst:
            break
        instance = stream.next_instance()
        pred = clf.predict(instance)
        y_true.append(int(instance.y_index))
        y_pred.append(int(pred) if pred is not None else 0)
        clf.train(instance)
        count += 1

    elapsed = time.time() - t0
    metrics = compute_metrics(y_true, y_pred)

    # Memory
    mem_kb = 0
    try:
        from pympler import asizeof
        mem_kb = asizeof.asizeof(clf) / 1024
    except ImportError:
        pass

    print(f"\n{name}")
    print(f"  Acc={metrics['accuracy']:.4f}  "
          f"BalAcc={metrics['balanced_accuracy']:.4f}  "
          f"MacF1={metrics['macro_f1']:.4f}  "
          f"TScore={metrics['tournament_score']:.4f}  "
          f"Mem={mem_kb:.1f}KB  "
          f"Time={elapsed:.1f}s")
    print(f"  Per-class recall: {metrics['per_class_recall']}")

    return metrics, mem_kb, elapsed


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "sample_test_csv.csv")

    if not os.path.exists(csv_path):
        print(f"ERROR: Sample CSV not found: {csv_path}")
        sys.exit(1)

    print("=" * 60)
    print("Quick Test — Sample Data (37,440 instances)")
    print("=" * 60)

    # Get schema
    stream = create_csv_stream(csv_path)
    schema = stream.get_schema()

    # 1. NaiveBayes baseline
    test_classifier(
        csv_path, schema,
        lambda s: NaiveBayes(schema=s),
        "NaiveBayes (baseline)"
    )

    # 2. Base Ensemble (programming task)
    test_classifier(
        csv_path, schema,
        lambda s: EnsembleClassifier(schema=s, n_members=10),
        "EnsembleClassifier (n=10)"
    )

    # 3. Champion (tournament)
    test_classifier(
        csv_path, schema,
        lambda s: ChampionClassifier(schema=s, n_members=100),
        "ChampionClassifier (n=100)"
    )

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
