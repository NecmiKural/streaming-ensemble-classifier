"""
COMPX523-26A Assignment 2: Evaluation Script

Tests the ensemble classifiers against:
  1. Synthetic RBF drift streams (baseline comparison)
  2. Instructor's sample test data (CSV/ARFF files)

Metrics: Accuracy, Balanced Accuracy, Macro F1, Kappa
Output: Results CSV in tournament-compatible format
"""

import os
import sys
import time
import csv
import numpy as np

# Set JAVA_HOME before importing capymoa, but only if not already set
# to a valid path. This keeps the script portable across machines (and
# allows the tournament evaluator to use whatever Java is installed
# server-side).
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

from capymoa.stream.generator import RandomRBFGeneratorDrift, RandomRBFGenerator
from capymoa.stream.drift import DriftStream, AbruptDrift, GradualDrift
from capymoa.stream import CSVStream, ARFFStream
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import NaiveBayes, AdaptiveRandomForestClassifier

# Import our classifiers
from ensemble_classifier import EnsembleClassifier
from champion_classifier import ChampionClassifier


# ==============================================================================
# Stream Builders
# ==============================================================================

def create_tournament_like_stream():
    """
    Create a stream that mimics the tournament dataset characteristics:
    - RBF generator (instructor hint: "data generator is synthetic, using RBF")
    - 12 numerical features
    - 2-class problem
    - ~100,000 instances
    - 4 concept drifts spread roughly evenly
    """
    stream = DriftStream(
        stream=[
            RandomRBFGeneratorDrift(
                model_random_seed=1,
                instance_random_seed=1,
                number_of_classes=2,
                number_of_attributes=12,
                number_of_centroids=20,
                magnitude_of_change=0.0,
            ),
            AbruptDrift(position=25000),
            RandomRBFGeneratorDrift(
                model_random_seed=2,
                instance_random_seed=2,
                number_of_classes=2,
                number_of_attributes=12,
                number_of_centroids=20,
                magnitude_of_change=0.0,
            ),
            GradualDrift(position=50000, width=864),
            RandomRBFGeneratorDrift(
                model_random_seed=3,
                instance_random_seed=3,
                number_of_classes=2,
                number_of_attributes=12,
                number_of_centroids=20,
                magnitude_of_change=0.0,
            ),
            GradualDrift(position=75000, width=2016),
            RandomRBFGeneratorDrift(
                model_random_seed=4,
                instance_random_seed=4,
                number_of_classes=2,
                number_of_attributes=12,
                number_of_centroids=20,
                magnitude_of_change=0.0,
            ),
        ]
    )
    return stream


def create_simple_rbf_stream():
    """Simpler RBF stream for quick testing."""
    stream = DriftStream(
        stream=[
            RandomRBFGeneratorDrift(
                model_random_seed=1,
                instance_random_seed=1,
                number_of_classes=2,
                number_of_attributes=12,
                number_of_centroids=15,
                magnitude_of_change=0.0,
            ),
            AbruptDrift(position=10000),
            RandomRBFGeneratorDrift(
                model_random_seed=2,
                instance_random_seed=2,
                number_of_classes=2,
                number_of_attributes=12,
                number_of_centroids=15,
                magnitude_of_change=0.0,
            ),
        ]
    )
    return stream


def create_csv_stream(csv_path):
    """
    Create a stream from the instructor's sample CSV file.
    Format: att1,att2,...,att12,class (with class values {0, 1})
    """
    stream = CSVStream(
        file=csv_path,
        target="class",
        categories={"class": ["0", "1"]},
        name="sample_test",
        length=37440,
    )
    return stream


def create_arff_stream(arff_path):
    """
    Create a stream from the instructor's sample ARFF file.
    """
    stream = ARFFStream(path=arff_path)
    return stream


# ==============================================================================
# Manual Prequential Evaluation (for custom metrics)
# ==============================================================================

def manual_prequential_eval(stream, learner, name, max_instances=None):
    """
    Custom prequential evaluation that tracks per-instance predictions
    for computing Balanced Accuracy and Macro F1.

    Returns a dict with all metrics.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    y_true = []
    y_pred = []
    instance_count = 0
    start_time = time.time()

    while stream.has_more_instances():
        if max_instances and instance_count >= max_instances:
            break

        instance = stream.next_instance()

        # Test-then-train (prequential)
        prediction = learner.predict(instance)
        y_true.append(int(instance.y_index))
        y_pred.append(int(prediction) if prediction is not None else 0)

        # Train
        learner.train(instance)
        instance_count += 1

        if instance_count % 5000 == 0:
            print(f"  ... {instance_count} instances processed")

    elapsed = time.time() - start_time

    # Compute metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    # Accuracy
    accuracy = np.mean(y_true == y_pred)

    # Per-class recall (for balanced accuracy)
    recalls = []
    precisions = []
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    balanced_accuracy = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)

    # Kappa
    total = len(y_true)
    p_observed = accuracy
    p_expected = sum(
        (np.sum(y_true == c) / total) * (np.sum(y_pred == c) / total)
        for c in classes
    )
    kappa = (p_observed - p_expected) / (1 - p_expected) if (1 - p_expected) > 0 else 0.0

    # Tournament score (weighted sum as described by instructor)
    # Using equal weighting as default since exact weights weren't specified
    tournament_score = 0.5 * balanced_accuracy + 0.5 * macro_f1

    # Class distribution
    class_dist = {int(c): int(np.sum(y_true == c)) for c in classes}

    results = {
        'name': name,
        'instances': instance_count,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'kappa': kappa,
        'tournament_score': tournament_score,
        'time': elapsed,
        'class_distribution': class_dist,
        'per_class_recall': {int(c): r for c, r in zip(classes, recalls)},
        'per_class_f1': {int(c): f for c, f in zip(classes, f1s)},
    }

    print(f"\nResults for {name}:")
    print(f"  Instances:          {instance_count:,}")
    print(f"  Accuracy:           {accuracy:.4f}")
    print(f"  Balanced Accuracy:  {balanced_accuracy:.4f}")
    print(f"  Macro F1:           {macro_f1:.4f}")
    print(f"  Kappa:              {kappa:.4f}")
    print(f"  Tournament Score:   {tournament_score:.4f}")
    print(f"  Wall-clock time:    {elapsed:.2f}s")
    print(f"  Class distribution: {class_dist}")
    print(f"  Per-class recall:   {results['per_class_recall']}")
    print(f"  Per-class F1:       {results['per_class_f1']}")

    return results


# ==============================================================================
# Memory Profiling
# ==============================================================================

def measure_memory(obj, name="Object"):
    """Measure memory usage.

    Reports two complementary numbers:

      * `pympler.asizeof` -- the Python-side reachable size. This is
        what the assignment PDF suggests in §3.2, but it misses MOA's
        Java-heap state for `HoeffdingTree`-backed objects (the actual
        tree nodes live on the JVM heap, invisible to pympler).

      * `obj.measure_java_byte_size()` -- when the classifier exposes
        it, the sum of MOA's own `measureByteSize()` across every base
        learner. This is the number MOA enforces `max_byte_size`
        against, and it's the correct estimate for HoeffdingTree-based
        ensembles.

    The instructor's tournament evaluator measures the Java-side
    footprint, so portfolio-quality submissions should pass that check
    rather than relying on pympler alone.
    """
    pympler_size = 0
    try:
        from pympler import asizeof
        pympler_size = asizeof.asizeof(obj)
    except ImportError:
        print("pympler not installed, skipping pympler measurement")

    java_size = 0
    if hasattr(obj, "measure_java_byte_size"):
        try:
            java_size = obj.measure_java_byte_size()
        except Exception as exc:  # noqa: BLE001
            print(f"  (Java byte-size probe failed: {exc})")

    total_effective = max(pympler_size, java_size + 30_000)  # +30 KB Python+ADWIN overhead

    print(f"\n{'='*60}")
    print(f"Memory Usage for {name}:")
    print(f"  pympler.asizeof   = {pympler_size:>9,} bytes "
          f"({pympler_size/1024:6.1f} KB)  [Python side only]")
    if java_size:
        print(f"  MOA Java trees    = {java_size:>9,} bytes "
              f"({java_size/1024:6.1f} KB)  [JVM heap, real footprint]")
        print(f"  Effective total   ≈ {total_effective:>9,} bytes "
              f"({total_effective/1024:6.1f} KB)  [Java + ~30 KB Python overhead]")
    print(f"{'='*60}")

    check_value = total_effective if java_size else pympler_size
    if check_value > 256000:
        over_pct = (check_value - 256000) / 256000 * 100
        print(f"⚠️  WARNING: {check_value/1024:.1f} KB exceeds 250 KB limit "
              f"by {over_pct:.1f}%")
    else:
        print(f"✅ Within 250 KB limit ({(256000 - check_value)/1024:.1f} KB remaining)")

    # Attribute-level breakdown (pympler only -- for the Python attributes)
    if pympler_size:
        from pympler import asizeof
        print(f"\nPython attribute breakdown (`pympler.asizeof`):")
        for attr, value in obj.__dict__.items():
            if attr.startswith('_'):
                continue
            attr_size = asizeof.asizeof(value)
            print(f"  {attr}: {attr_size:,} bytes ({attr_size/1024:.1f} KB)")

    return check_value


# ==============================================================================
# Results CSV Output
# ==============================================================================

def write_results_csv(results_list, output_path):
    """
    Write evaluation results to a CSV file in tournament format.
    """
    fieldnames = [
        'classifier', 'instances', 'accuracy', 'balanced_accuracy',
        'macro_f1', 'kappa', 'tournament_score', 'time_seconds', 'memory_kb'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_list:
            writer.writerow({
                'classifier': r.get('name', ''),
                'instances': r.get('instances', 0),
                'accuracy': f"{r.get('accuracy', 0):.6f}",
                'balanced_accuracy': f"{r.get('balanced_accuracy', 0):.6f}",
                'macro_f1': f"{r.get('macro_f1', 0):.6f}",
                'kappa': f"{r.get('kappa', 0):.6f}",
                'tournament_score': f"{r.get('tournament_score', 0):.6f}",
                'time_seconds': f"{r.get('time', 0):.2f}",
                'memory_kb': f"{r.get('memory_kb', 0):.1f}",
            })

    print(f"\n📄 Results saved to: {output_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 60)
    print("COMPX523-26A Assignment 2 — Full Evaluation")
    print("=" * 60)

    all_results = []
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ================================================================
    # Phase 1: Sample Data Test (instructor's CSV)
    # ================================================================
    sample_csv = os.path.join(script_dir, "sample_test_csv.csv")
    sample_arff = os.path.join(script_dir, "sample_arff_file.arff")

    if os.path.exists(sample_csv):
        print("\n\n>>> Phase 1: Sample Test Data (37,440 instances)")
        print("-" * 60)

        # 1a. NaiveBayes baseline
        stream = create_csv_stream(sample_csv)
        schema = stream.get_schema()
        nb = NaiveBayes(schema=schema)
        r_nb = manual_prequential_eval(stream, nb, "NaiveBayes (baseline)")
        all_results.append(r_nb)

        # 1b. Our base ensemble (programming task)
        stream = create_csv_stream(sample_csv)
        ens = EnsembleClassifier(schema=schema, n_members=10)
        r_ens = manual_prequential_eval(stream, ens, "EnsembleClassifier (n=10)")
        mem_ens = measure_memory(ens, "EnsembleClassifier (n=10)")
        r_ens['memory_kb'] = mem_ens / 1024
        all_results.append(r_ens)

        # 1c. Our champion (uses class defaults: n=25, 8KB/tree, mem_period=1000)
        stream = create_csv_stream(sample_csv)
        champ = ChampionClassifier(schema=schema)
        r_champ = manual_prequential_eval(stream, champ, str(champ))
        mem_champ = measure_memory(champ, str(champ))
        r_champ['memory_kb'] = mem_champ / 1024
        all_results.append(r_champ)

    else:
        print(f"\n⚠️  Sample CSV not found: {sample_csv}")
        print("Skipping sample data test. Using synthetic RBF stream only.")

    # ================================================================
    # Phase 2: Synthetic RBF Test (with drift)
    # ================================================================
    print("\n\n>>> Phase 2: Synthetic RBF Test (100,000 instances, 4 drifts)")
    print("-" * 60)

    stream = create_tournament_like_stream()
    schema = stream.get_schema()

    # 2a. NaiveBayes baseline
    stream = create_tournament_like_stream()
    nb_synth = NaiveBayes(schema=schema)
    r_nb_s = manual_prequential_eval(stream, nb_synth, "NaiveBayes (synthetic)", max_instances=100000)
    all_results.append(r_nb_s)

    # 2b. Our base ensemble
    stream = create_tournament_like_stream()
    ens_synth = EnsembleClassifier(schema=schema, n_members=10)
    r_ens_s = manual_prequential_eval(stream, ens_synth, "EnsembleClassifier (synthetic)", max_instances=100000)
    all_results.append(r_ens_s)

    # 2c. Our champion
    stream = create_tournament_like_stream()
    champ_synth = ChampionClassifier(schema=schema)
    r_champ_s = manual_prequential_eval(
        stream, champ_synth,
        f"{champ_synth} (synthetic)",
        max_instances=100000,
    )
    mem_champ_s = measure_memory(champ_synth, f"{champ_synth} (synthetic)")
    r_champ_s['memory_kb'] = mem_champ_s / 1024
    all_results.append(r_champ_s)

    # ================================================================
    # Summary
    # ================================================================
    print("\n\n>>> SUMMARY")
    print("=" * 100)
    header = f"{'Classifier':<45} {'Acc':>7} {'BalAcc':>7} {'MacF1':>7} {'Kappa':>7} {'TScore':>7} {'Time':>8}"
    print(header)
    print("-" * 100)
    for r in all_results:
        line = (
            f"{r['name']:<45} "
            f"{r['accuracy']:>7.4f} "
            f"{r['balanced_accuracy']:>7.4f} "
            f"{r['macro_f1']:>7.4f} "
            f"{r['kappa']:>7.4f} "
            f"{r['tournament_score']:>7.4f} "
            f"{r['time']:>7.2f}s"
        )
        print(line)
    print("=" * 100)

    # Write results CSV
    results_csv_path = os.path.join(script_dir, "results.csv")
    write_results_csv(all_results, results_csv_path)


if __name__ == "__main__":
    main()
