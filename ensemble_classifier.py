"""
COMPX523-26A Assignment 2: Ensemble Classifier for Data Streams
Author: Necmi Kural

This module implements the Programming Task (10%) ensemble classifier with:
  1. Online Bagging (diversity generation via Poisson(λ=1))           — §2.1
  2. Gaussian Naive Bayes base learners (incremental, Welford update) — §2.2
  3. ADWIN drift detection (per-member monitoring + reset)            — §2.3
  4. Leveraged Bagging for class balancing (dynamic λ via window)     — §2.4

ADWIN integration uses CapyMOA's built-in MOA-backed ADWIN
(`capymoa.drift.detectors.ADWIN`) per assignment spec §2.3:
    "Each ensemble member should be monitored by an ADWIN instance."
"""

import math

import numpy as np
from capymoa.base import Classifier
from capymoa.drift.detectors import ADWIN

# ==============================================================================
# Incremental Gaussian Naive Bayes Base Learner
# ==============================================================================
class IncrementalGNB:
    """
    Memory-efficient Gaussian Naive Bayes for streaming data.
    
    Uses Welford's numerically stable online algorithm for incremental
    mean and variance updates. Stores only sufficient statistics:
    (weight_sum, mean, variance_sum) per feature per class.
    """

    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        self.n_classes = n_classes

        # Sufficient statistics per class per feature
        # Shape: [n_classes, n_features]
        self.weight_sum = np.zeros((n_classes, n_features), dtype=np.float64)
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.variance_sum = np.zeros((n_classes, n_features), dtype=np.float64)

        # Class priors (total weight per class)
        self.class_weight = np.zeros(n_classes, dtype=np.float64)
        self.total_weight = 0.0

    def reset(self):
        """Reset all statistics (called when ADWIN detects drift)."""
        self.weight_sum[:] = 0.0
        self.mean[:] = 0.0
        self.variance_sum[:] = 0.0
        self.class_weight[:] = 0.0
        self.total_weight = 0.0

    def train(self, features, label, weight=1.0):
        """
        Incrementally update the model with a single instance.
        
        Uses Welford's online algorithm (Algorithm 4 from the textbook):
            mean_new = mean_old + (x - mean_old) / weight_sum_new
            variance_sum += (x - mean_old) * (x - mean_new)
        
        Args:
            features: numpy array of feature values
            label: integer class label
            weight: instance weight (from Poisson sampling)
        """
        c = int(label)
        for _ in range(int(weight)):
            for j in range(self.n_features):
                x = features[j]
                self.weight_sum[c, j] += 1.0
                old_mean = self.mean[c, j]
                self.mean[c, j] += (x - old_mean) / self.weight_sum[c, j]
                self.variance_sum[c, j] += (x - old_mean) * (x - self.mean[c, j])

            self.class_weight[c] += 1.0
            self.total_weight += 1.0

    def predict(self, features):
        """
        Predict the class label using MAP estimation.
        
        For each class c:
            P(c|x) ∝ P(c) * Π P(x_j | c)
        where P(x_j | c) = N(x_j; mean_cj, var_cj)
        
        Uses log-probabilities to avoid numerical underflow.
        
        Returns:
            Predicted class label (int)
        """
        if self.total_weight == 0:
            return 0  # no data seen yet

        best_class = 0
        best_log_prob = -math.inf

        for c in range(self.n_classes):
            if self.class_weight[c] == 0:
                continue

            # Log prior: log P(c)
            log_prob = math.log(self.class_weight[c] / self.total_weight)

            for j in range(self.n_features):
                if self.weight_sum[c, j] < 2:
                    continue  # need at least 2 samples for variance

                variance = self.variance_sum[c, j] / (self.weight_sum[c, j] - 1)
                if variance < 1e-10:
                    variance = 1e-10  # avoid division by zero

                x = features[j]
                mu = self.mean[c, j]

                # Log of Gaussian PDF: -0.5 * [log(2π) + log(σ²) + (x-μ)²/σ²]
                log_prob += -0.5 * (math.log(2 * math.pi * variance) + 
                                     (x - mu) ** 2 / variance)

            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = c

        return best_class

    def predict_proba(self, features):
        """
        Return class probabilities using softmax over log-probabilities.
        """
        if self.total_weight == 0:
            return np.ones(self.n_classes) / self.n_classes

        log_probs = np.full(self.n_classes, -math.inf)

        for c in range(self.n_classes):
            if self.class_weight[c] == 0:
                continue

            log_prob = math.log(self.class_weight[c] / self.total_weight)

            for j in range(self.n_features):
                if self.weight_sum[c, j] < 2:
                    continue

                variance = self.variance_sum[c, j] / (self.weight_sum[c, j] - 1)
                if variance < 1e-10:
                    variance = 1e-10

                x = features[j]
                mu = self.mean[c, j]
                log_prob += -0.5 * (math.log(2 * math.pi * variance) + 
                                     (x - mu) ** 2 / variance)

            log_probs[c] = log_prob

        # Softmax with numerical stability
        max_log = np.max(log_probs)
        if max_log == -math.inf:
            return np.ones(self.n_classes) / self.n_classes
        
        exp_probs = np.exp(log_probs - max_log)
        return exp_probs / np.sum(exp_probs)


# ==============================================================================
# Main Ensemble Classifier
# ==============================================================================
class EnsembleClassifier(Classifier):
    """
    Ensemble Classifier for Data Streams with:
    - Online Bagging (Poisson(λ=1) diversity generation)
    - Gaussian Naive Bayes base learners
    - ADWIN-based drift detection per member
    - Leveraged Bagging for class imbalance handling
    
    COMPX523-26A Assignment 2 — Programming Task
    """

    def __init__(self, schema, n_members=10, lambda_base=1, lambda_leverage=6,
                 adwin_delta=0.002, class_window_size=500):
        """
        Args:
            schema: CapyMOA stream schema
            n_members: Number of ensemble members
            lambda_base: Base Poisson λ for majority class (Online Bagging)
            lambda_leverage: Poisson λ ceiling for minority class (Leveraged Bagging)
            adwin_delta: ADWIN confidence parameter (smaller = less sensitive)
            class_window_size: Window size for tracking class distribution
        """
        super().__init__(schema)
        self.n_members = n_members
        self.lambda_base = lambda_base
        self.lambda_leverage = lambda_leverage
        self.adwin_delta = adwin_delta
        self.class_window_size = class_window_size

        n_features = schema.get_num_attributes()
        n_classes = schema.get_num_classes()

        # Initialize ensemble members (Gaussian Naive Bayes — §2.2)
        self.base_learners = [
            IncrementalGNB(n_features, n_classes)
            for _ in range(n_members)
        ]

        # Initialize ADWIN drift detectors — one per ensemble member (§2.3)
        # Using CapyMOA's MOA-backed ADWIN: capymoa.drift.detectors.ADWIN
        self.drift_detectors = [
            ADWIN(delta=adwin_delta)
            for _ in range(n_members)
        ]

        # Class distribution tracking window for dynamic λ (Leveraged Bagging — §2.4)
        self.class_counts = np.zeros(n_classes, dtype=np.float64)
        self.class_history = []  # sliding window of recent labels

        self.n_features = n_features
        self.n_classes = n_classes
        self._rng = np.random.default_rng(42)

        # Counter for periodic ADWIN history cleanup. The MOA-backed ADWIN
        # in CapyMOA appends every observation to a Python `data` list for
        # history bookkeeping. The list isn't used by the Java change-detection
        # algorithm, but it grows unbounded — so we periodically purge it.
        self._instances_seen = 0
        self._history_purge_every = 1000

    def __str__(self):
        return f"EnsembleClassifier(n={self.n_members})"

    def _get_lambda(self, label):
        """
        Determine Poisson λ based on whether the instance is from the 
        minority or majority class.
        
        Uses a sliding window of recent labels to estimate class distribution.
        The minority class gets a higher λ (Leveraged Bagging) to combat
        class imbalance.
        """
        if len(self.class_history) < 10:
            return self.lambda_base

        # Count class frequencies in the recent window
        total = len(self.class_history)
        label_count = sum(1 for l in self.class_history if l == label)
        label_ratio = label_count / total

        # If this class makes up less than 40% of recent data, it's minority
        if label_ratio < 0.4:
            # Scale λ proportionally to the imbalance
            imbalance_ratio = (1 - label_ratio) / max(label_ratio, 0.01)
            return min(self.lambda_leverage * imbalance_ratio / 2, 
                       self.lambda_leverage * 2)
        else:
            return self.lambda_base

    def train(self, instance):
        """
        Train the ensemble on a new instance using Online Bagging.
        
        For each base learner:
            1. Sample k ~ Poisson(λ), where λ depends on class balance
            2. Train the base learner k times with the instance
            3. Check prediction error and feed to ADWIN
            4. If ADWIN detects drift → reset that base learner
        """
        features = np.array(instance.x)
        label = instance.y_index

        # Update class distribution window
        self.class_history.append(label)
        if len(self.class_history) > self.class_window_size:
            self.class_history.pop(0)

        # Determine λ for this instance's class
        lam = self._get_lambda(label)

        for i in range(self.n_members):
            # Online Bagging (§2.1): sample weight k ~ Poisson(λ),
            # then train the base learner k times with this instance.
            k = self._rng.poisson(lam)

            if k > 0:
                self.base_learners[i].train(features, label, weight=k)

            # Pre-train (test-then-train style) prediction is used only as
            # an error signal for ADWIN. We compute it AFTER training so the
            # error reflects this learner's current state.
            prediction = self.base_learners[i].predict(features)
            error = 0.0 if prediction == label else 1.0

            # ADWIN drift detection (§2.3) — feed the 0/1 error stream.
            # CapyMOA ADWIN API: add_element(value) then detected_change().
            self.drift_detectors[i].add_element(error)

            if self.drift_detectors[i].detected_change():
                # Concept drift detected on this member → re-initialize
                # both the base learner and the detector.
                self.base_learners[i].reset()
                self.drift_detectors[i].reset(clean_history=True)

        # Periodic memory hygiene: clear the Python-side history that
        # CapyMOA's MOADriftDetector base class records on every
        # add_element call. The MOA Java algorithm does NOT need this
        # list — it's only kept for inspection — so dropping it is safe.
        self._instances_seen += 1
        if self._instances_seen % self._history_purge_every == 0:
            for det in self.drift_detectors:
                det.data = []
                det.idx = 0

    def predict(self, instance):
        """
        Predict class label using majority vote across ensemble members.
        """
        features = np.array(instance.x)
        votes = np.zeros(self.n_classes, dtype=np.int32)

        for learner in self.base_learners:
            pred = learner.predict(features)
            votes[pred] += 1

        return int(np.argmax(votes))

    def predict_proba(self, instance):
        """
        Average probability predictions across ensemble members.
        """
        features = np.array(instance.x)
        avg_proba = np.zeros(self.n_classes, dtype=np.float64)

        for learner in self.base_learners:
            avg_proba += learner.predict_proba(features)

        avg_proba /= self.n_members
        return avg_proba
