"""
Quick experiment: Hoeffding Tree vs Gaussian NB as base learner.

Tests an HT-backed Champion variant on the instructor's sample CSV.
Tries multiple (n_members, max_byte_size) configurations to find the
sweet spot between accuracy and the 250 KB memory budget.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Iterable

import numpy as np
from pympler import asizeof

os.environ.setdefault(
    "JAVA_HOME", "/opt/homebrew/Cellar/openjdk@17/17.0.18/libexec/openjdk.jdk/Contents/Home"
)

from capymoa.base import Classifier
from capymoa.classifier import HoeffdingTree
from capymoa.drift.detectors import ADWIN
from capymoa.stream import CSVStream
from capymoa.stream._stream import Schema


SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "sample_test_csv.csv")


class HTChampion(Classifier):
    """
    Champion variant using Hoeffding Tree base learners.

    Components:
      - HoeffdingTree base learners (constrained max_byte_size)
      - Online Bagging via Poisson(λ) per learner
      - Per-member ADWIN drift detection + reset
      - Leveraged Bagging: dynamic λ from class ring buffer
      - Periodic ADWIN history purge for memory hygiene
    """

    def __init__(
        self,
        schema: Schema,
        n_members: int = 25,
        max_byte_size: int = 8 * 1024,
        lambda_base: float = 1.0,
        lambda_leverage: float = 2.0,
        adwin_delta: float = 0.002,
        class_window_size: int = 2000,
        history_purge_every: int = 1000,
        random_seed: int = 42,
    ) -> None:
        super().__init__(schema)
        self.n_members = n_members
        self.max_byte_size = max_byte_size
        self.lambda_base = lambda_base
        self.lambda_leverage = lambda_leverage
        self.adwin_delta = adwin_delta
        self.class_window_size = class_window_size
        self.history_purge_every = history_purge_every
        self.random_seed = random_seed

        self.n_classes = schema.get_num_classes()
        self._schema = schema

        self.base_learners = [self._new_tree(seed=random_seed + i) for i in range(n_members)]
        self.drift_detectors = [ADWIN(delta=adwin_delta) for _ in range(n_members)]

        self._class_ring = np.zeros(class_window_size, dtype=np.int8)
        self._ring_pos = 0
        self._ring_count = 0

        self._rng = np.random.default_rng(random_seed)
        self._instances_seen = 0

    def _new_tree(self, seed: int) -> HoeffdingTree:
        return HoeffdingTree(
            schema=self._schema,
            random_seed=seed,
            max_byte_size=self.max_byte_size,
            grace_period=50,         # split decisions sooner on streams
            confidence=0.05,         # looser Hoeffding bound → tighter trees
            tie_threshold=0.05,
            leaf_prediction="NaiveBayesAdaptive",
            stop_mem_management=True,
        )

    def __str__(self) -> str:
        return f"HTChampion(n={self.n_members}, byte_size={self.max_byte_size})"

    def _get_lambda(self, label: int) -> float:
        if self._ring_count < 20:
            return self.lambda_base
        n = min(self._ring_count, self.class_window_size)
        label_count = int(np.sum(self._class_ring[:n] == label))
        label_ratio = label_count / n
        if label_ratio < 0.35:
            imbalance = (1.0 - label_ratio) / max(label_ratio, 0.01)
            return min(self.lambda_leverage * imbalance / 2.0, 8.0)
        return self.lambda_base

    def train(self, instance) -> None:
        label = int(instance.y_index)
        self._class_ring[self._ring_pos] = label
        self._ring_pos = (self._ring_pos + 1) % self.class_window_size
        self._ring_count += 1

        lam = self._get_lambda(label)
        weights = self._rng.poisson(lam, size=self.n_members)

        for i in range(self.n_members):
            k = int(weights[i])
            if k > 0:
                # CapyMOA HoeffdingTree exposes `train(instance)` — re-train k times.
                for _ in range(k):
                    self.base_learners[i].train(instance)

            pred = self.base_learners[i].predict(instance)
            error = 0.0 if pred == label else 1.0
            self.drift_detectors[i].add_element(error)

            if self.drift_detectors[i].detected_change():
                self.base_learners[i] = self._new_tree(
                    seed=self.random_seed + i + self._instances_seen
                )
                self.drift_detectors[i].reset(clean_history=True)

        self._instances_seen += 1
        if self._instances_seen % self.history_purge_every == 0:
            for det in self.drift_detectors:
                det.data = []
                det.idx = 0

    def predict(self, instance) -> int:
        votes = np.zeros(self.n_classes, dtype=np.int32)
        for learner in self.base_learners:
            pred = learner.predict(instance)
            if pred is None:
                continue
            votes[int(pred)] += 1
        return int(np.argmax(votes))

    def predict_proba(self, instance) -> np.ndarray:
        avg = np.zeros(self.n_classes, dtype=np.float64)
        n_seen = 0
        for learner in self.base_learners:
            proba = learner.predict_proba(instance)
            if proba is None:
                continue
            avg += np.asarray(proba, dtype=np.float64)
            n_seen += 1
        if n_seen == 0:
            return np.ones(self.n_classes) / self.n_classes
        return avg / n_seen


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    acc = float(np.mean(y_true == y_pred))
    recalls, f1s = [], []
    for c in classes:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        recalls.append(rec)
        f1s.append(f1)
    bal = float(np.mean(recalls))
    macf1 = float(np.mean(f1s))
    return {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "macro_f1": macf1,
        "tournament_score": 0.5 * bal + 0.5 * macf1,
        "per_class_recall": {int(c): float(r) for c, r in zip(classes, recalls)},
    }


def evaluate(clf: Classifier, csv_path: str, name: str, max_n: int | None = None) -> dict:
    stream = CSVStream(
        file=csv_path, target="class",
        categories={"class": ["0", "1"]}, name="sample", length=37440,
    )
    y_true: list[int] = []
    y_pred: list[int] = []
    t0 = time.time()
    count = 0
    while stream.has_more_instances():
        if max_n is not None and count >= max_n:
            break
        inst = stream.next_instance()
        y_true.append(int(inst.y_index))
        y_pred.append(int(clf.predict(inst)))
        clf.train(inst)
        count += 1
    elapsed = time.time() - t0
    m = metrics(np.asarray(y_true), np.asarray(y_pred))
    mem_kb = asizeof.asizeof(clf) / 1024
    m.update({"name": name, "time_s": elapsed, "memory_kb": mem_kb, "n_instances": count})
    print(
        f"  {name:<45} Acc={m['accuracy']:.3f}  BalAcc={m['balanced_accuracy']:.3f}  "
        f"MacF1={m['macro_f1']:.3f}  TScore={m['tournament_score']:.3f}  "
        f"Mem={mem_kb:6.1f}KB  Time={elapsed:6.1f}s  "
        f"R={m['per_class_recall']}"
    )
    return m


def configurations() -> Iterable[tuple[int, int]]:
    """(n_members, max_byte_size) sweep — chosen to stay within 250 KB."""
    return [
        (10, 4 * 1024),
        (10, 8 * 1024),
        (25, 4 * 1024),
        (25, 8 * 1024),
        (50, 2 * 1024),
        (50, 4 * 1024),
    ]


def main() -> None:
    stream = CSVStream(
        file=SAMPLE_CSV, target="class",
        categories={"class": ["0", "1"]}, name="sample", length=37440,
    )
    schema = stream.get_schema()

    print("=" * 100)
    print("Hoeffding Tree Champion Variant — Sweep on sample_test_csv.csv (37,440 instances)")
    print("=" * 100)

    results = []
    for n_members, max_byte in configurations():
        clf = HTChampion(schema=schema, n_members=n_members, max_byte_size=max_byte)
        name = f"HT(n={n_members}, byte_size={max_byte//1024}KB)"
        # Use a 15 K sub-stream for fast sweep
        results.append(evaluate(clf, SAMPLE_CSV, name, max_n=15000))

    print()
    print("Best by TScore:")
    for r in sorted(results, key=lambda x: -x["tournament_score"])[:3]:
        print(f"  {r['name']:<45} TScore={r['tournament_score']:.3f} Mem={r['memory_kb']:.1f}KB Time={r['time_s']:.1f}s")


if __name__ == "__main__":
    main()
