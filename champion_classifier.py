"""
COMPX523-26A Assignment 2: Champion Classifier for Tournament
Author: Necmi Kural

Swaps the spec's default GNB base learner for a memory-constrained
Hoeffding Tree. GNB plateaued around TScore 0.61 on the released
sample data because the RBF generator's mixture-of-Gaussians decision
surface violates GNB's single-Gaussian-per-class assumption.

POST-SUBMISSION FIX (portfolio version)
---------------------------------------
The originally submitted version had a subtle memory bug that this
revision corrects:

  * `max_byte_size` on a CapyMOA `HoeffdingTree` is enforced ONLY
    when MOA's periodic memory estimation runs, which by default is
    every `memory_estimate_period = 1_000_000` instances. For streams
    shorter than that (the assignment sample stream has ~37 K, the
    synthetic 100 K), the check never fired and trees grew freely.
  * `pympler.asizeof` only measures the Python wrapper object, not
    the underlying MOA Java heap state, so our local measurement of
    "156 KB" was an undercount. The instructor's evaluator (which
    sees the real Java heap) reported ~25 MB for the original n=100
    configuration.

The fix is two-fold:
  1. Pass `memory_estimate_period=1000` so MOA actually enforces
     `max_byte_size` early and often.
  2. Sanity-check memory via MOA's own `measureByteSize()` Java
     method (see `measure_java_byte_size()` below) instead of
     trusting `pympler.asizeof` alone.

Components:
  1. Hoeffding Tree base learners (capymoa.classifier.HoeffdingTree)
     with `max_byte_size` and `memory_estimate_period` set so the per-
     tree footprint is genuinely bounded.
  2. Online Bagging via Poisson(λ) per learner (PDF §2.1).
  3. Per-member ADWIN drift detection (capymoa.drift.detectors.ADWIN),
     with re-init on detected change (PDF §2.3). The detector is
     wrapped in an `_ADWINNoHistory` subclass that skips the unbounded
     Python-side observation list (see the subclass docstring).
  4. Leveraged Bagging with a dynamic λ derived from a class ring
     buffer to handle the 2:1–8:1 imbalance (PDF §2.4).
"""

from __future__ import annotations

import numpy as np
from capymoa.base import Classifier
from capymoa.classifier import HoeffdingTree
from capymoa.drift.detectors import ADWIN
from numpy import double as _np_double


class _ADWINNoHistory(ADWIN):
    """ADWIN subclass that does not keep a Python-side observation log.

    The CapyMOA `MOADriftDetector` base class appends every value passed
    to ``add_element`` into ``self.data`` for user inspection. Across a
    100-member ensemble × 100 K instances this list dominates the memory
    profile (measured ≈ 446 KB on sample data alone). The actual
    change-detection algorithm runs on the MOA Java side and never reads
    this list, so disabling the append is safe and keeps us inside the
    250 KB tournament budget.
    """

    def add_element(self, element: float) -> None:  # type: ignore[override]
        # Mirror the parent implementation MINUS the bookkeeping list.
        self.moa_detector.input(_np_double(element))
        self.idx += 1


class ChampionClassifier(Classifier):
    """Tournament Champion for COMPX523-26A Assignment 2.

    Portfolio version (post-submission). Defaults sized so the actual
    Java-heap memory footprint stays well within the 250 KB cap when
    `memory_estimate_period=1000` actually enforces it. The instructor's
    reference HoeffdingAdaptiveTree(n=10) reached 0.88 TScore at
    112 KB total memory, which is the regime this version targets.

        n_members=25, max_byte_size=8 KB, memory_estimate_period=1000,
        λ_leverage=2.0, adwin_delta=0.002

    Approximate budget at the chosen defaults:
        25 trees × 8 KB cap     = 200 KB worst case
      + 25 ADWIN (no-history)   ≈  20 KB
      + class ring buffer       ≈   2 KB
      + schema + overhead       ≈   5 KB
      --------------------------------------------
      total worst case          ≈ 227 KB  (≤ 250 KB ✔)
    """

    def __init__(
        self,
        schema,
        n_members: int = 25,
        max_byte_size: int = 8 * 1024,
        memory_estimate_period: int = 1000,
        lambda_base: float = 1.0,
        lambda_leverage: float = 2.0,
        adwin_delta: float = 0.002,
        class_window_size: int = 2000,
        grace_period: int = 50,
        confidence: float = 0.05,
        random_seed: int = 42,
    ) -> None:
        super().__init__(schema)

        self.n_members = n_members
        self.max_byte_size = max_byte_size
        self.memory_estimate_period = memory_estimate_period
        self.lambda_base = lambda_base
        self.lambda_leverage = lambda_leverage
        self.adwin_delta = adwin_delta
        self.class_window_size = class_window_size
        self.grace_period = grace_period
        self.confidence = confidence
        self.random_seed = random_seed

        self.n_classes = schema.get_num_classes()
        self._schema = schema

        # Base learners — one Hoeffding Tree per ensemble member.
        # Each tree is seeded differently to ensure structural diversity
        # beyond what Online Bagging alone provides.
        self.base_learners = [
            self._new_tree(seed=random_seed + i) for i in range(n_members)
        ]

        # ADWIN drift detector per member (PDF §2.3).
        # Use the no-history subclass so the Python-side bookkeeping list
        # never grows — see `_ADWINNoHistory` docstring above.
        self.drift_detectors = [
            _ADWINNoHistory(delta=adwin_delta) for _ in range(n_members)
        ]

        # Class ring buffer for Leveraged Bagging's dynamic λ (PDF §2.4).
        self._class_ring = np.zeros(class_window_size, dtype=np.int8)
        self._ring_pos = 0
        self._ring_count = 0

        self._rng = np.random.default_rng(random_seed)
        self._instances_seen = 0

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _new_tree(self, seed: int) -> HoeffdingTree:
        """Create a single Hoeffding Tree configured for the streaming setting.

        Memory enforcement requires BOTH:
            * `max_byte_size`           -- the per-tree byte cap
            * `memory_estimate_period`  -- how often MOA actually
                                            checks against the cap.
              The CapyMOA default is 1,000,000, which means the cap
              never enforces on assignment-scale streams. Setting it
              to 1000 makes enforcement effectively continuous.
        """
        return HoeffdingTree(
            schema=self._schema,
            random_seed=seed,
            max_byte_size=self.max_byte_size,
            memory_estimate_period=self.memory_estimate_period,
            grace_period=self.grace_period,
            confidence=self.confidence,
            tie_threshold=0.05,
            leaf_prediction="NaiveBayesAdaptive",
            stop_mem_management=True,
        )

    def measure_java_byte_size(self) -> int:
        """Sum MOA's own `measureByteSize()` across every base tree.

        `pympler.asizeof` only sees the Python wrapper object, not the
        MOA Java heap state where the tree nodes actually live. MOA's
        `measureByteSize()` reports the Java footprint each tree
        accounts to its `max_byte_size` cap, so summing it across the
        ensemble gives the realistic per-tree memory total.
        """
        total = 0
        for learner in self.base_learners:
            moa = getattr(learner, "moa_learner", None)
            if moa is not None and hasattr(moa, "measureByteSize"):
                total += int(moa.measureByteSize())
        return total

    def __str__(self) -> str:
        return (
            f"ChampionClassifier("
            f"n={self.n_members}, byte={self.max_byte_size // 1024}KB, "
            f"λ_lev={self.lambda_leverage})"
        )

    # ------------------------------------------------------------------
    # Class-balancing helper
    # ------------------------------------------------------------------
    def _get_lambda(self, label: int) -> float:
        """Dynamic Poisson λ from the ring buffer.

        If the current instance belongs to a class that occupies less
        than 35 % of the recent window, oversample it via a higher λ
        capped at 8.0 (PDF §2.4: Leveraged Bagging). Otherwise use the
        base λ.
        """
        if self._ring_count < 20:
            return self.lambda_base

        n = min(self._ring_count, self.class_window_size)
        label_count = int(np.sum(self._class_ring[:n] == label))
        label_ratio = label_count / n

        if label_ratio < 0.35:
            imbalance = (1.0 - label_ratio) / max(label_ratio, 0.01)
            return min(self.lambda_leverage * imbalance / 2.0, 8.0)

        return self.lambda_base

    # ------------------------------------------------------------------
    # Online learning loop
    # ------------------------------------------------------------------
    def train(self, instance) -> None:
        label = int(instance.y_index)

        # Update class ring buffer (fixed-size, no allocations).
        self._class_ring[self._ring_pos] = label
        self._ring_pos = (self._ring_pos + 1) % self.class_window_size
        self._ring_count += 1

        # Determine λ and draw all Poisson weights at once for speed.
        lam = self._get_lambda(label)
        weights = self._rng.poisson(lam, size=self.n_members)

        for i in range(self.n_members):
            k = int(weights[i])
            if k > 0:
                # Online Bagging (PDF §2.1): present the instance k times.
                for _ in range(k):
                    self.base_learners[i].train(instance)

            # Error signal for ADWIN: 0 if correct, 1 if wrong.
            pred = self.base_learners[i].predict(instance)
            error = 0.0 if pred is not None and int(pred) == label else 1.0
            self.drift_detectors[i].add_element(error)

            if self.drift_detectors[i].detected_change():
                # Concept drift on this member → start a fresh tree.
                self.base_learners[i] = self._new_tree(
                    seed=self.random_seed + i + self._instances_seen
                )
                self.drift_detectors[i].reset(clean_history=True)

        # No periodic purge needed — `_ADWINNoHistory` never grows the
        # Python-side history list to begin with.
        self._instances_seen += 1

    def predict(self, instance) -> int:
        """Majority vote across ensemble members."""
        votes = np.zeros(self.n_classes, dtype=np.int32)
        for learner in self.base_learners:
            pred = learner.predict(instance)
            if pred is None:
                continue
            votes[int(pred)] += 1
        return int(np.argmax(votes))

    def predict_proba(self, instance) -> np.ndarray:
        """Average of per-learner posterior probabilities."""
        avg = np.zeros(self.n_classes, dtype=np.float64)
        seen = 0
        for learner in self.base_learners:
            proba = learner.predict_proba(instance)
            if proba is None:
                continue
            avg += np.asarray(proba, dtype=np.float64)
            seen += 1
        if seen == 0:
            return np.ones(self.n_classes) / self.n_classes
        return avg / seen
