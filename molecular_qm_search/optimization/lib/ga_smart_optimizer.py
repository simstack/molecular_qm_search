from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None


class SmartOptimizer:
    """
    SmartOptimizer is designed to assist in optimization tasks by leveraging
    machine learning to predict outcomes and determine whether certain
    calculations can be skipped based on predefined thresholds.

    The primary purpose of this class is to collect data, fit predictive
    models, and make informed decisions about whether to discard cases
    while optimizing computations.

    :ivar active: Indicates whether the optimizer is actively collecting
                  data and making predictions.
    :type active: bool
    :ivar samples: List of tuples representing collected optimization data
                   in the form (de_initial, e_initial, de_final).
    :type samples: list
    :ivar model: The trained predictive model for estimating outcomes,
                 if enough data samples are collected.
    :type model: Optional[LinearRegression]
    :ivar stats: Dictionary for storing various counters related to the
                 optimizer's performance, including total cases processed,
                 guessed outcomes, and skipped cases.
    :type stats: dict
    :ivar min_samples: Number of samples required before the optimizer
                       begins fitting the predictive model.
    :type min_samples: int
    """

    def __init__(self, active: bool = False):
        self.active = active
        self.samples = []  # List of (de_initial, e_initial, de_final)
        self.model = None
        self.stats = {"total": 0, "guessed": 0, "correct_discard": 0, "false_discard": 0, "skipped": 0}
        self.min_samples = 20

    def collect(self, de_initial: float, e_initial: float, de_final: float):
        if not self.active: return
        self.samples.append((de_initial, e_initial, de_final))
        if len(self.samples) >= self.min_samples and len(self.samples) % 10 == 0:
            self._fit()

    def _fit(self):
        if LinearRegression is None:
            return
        X = np.array([[s[0], s[1]] for s in self.samples])
        y = np.array([s[2] for s in self.samples])
        self.model = LinearRegression().fit(X, y)

    def estimate(self, de_initial: float, e_initial: float) -> Optional[Tuple[float, float]]:
        if not self.active or self.model is None:
            return None
        # Prediction
        pred = self.model.predict([[de_initial, e_initial]])[0]
        # Heuristic confidence based on R^2 and sample size
        r2 = self.model.score(np.array([[s[0], s[1]] for s in self.samples]),
                              np.array([s[2] for s in self.samples]))
        # We want > 80% confidence. Let's use R^2 as a proxy for model reliability
        # and check if the prediction is within expected range.
        if r2 > 0.8:
            return pred, r2
        return None

    def should_discard(self, de_initial: float, e_initial: float, energy_threshold: float) -> bool:
        if not self.active or self.model is None:
            return False

        est = self.estimate(de_initial, e_initial)
        if est is None: return False

        pred_de_final, confidence = est
        # Estimated final energy: initial_energy - (pred_de_final - de_initial)
        # Wait, de_initial = e_start - e_partial. e_partial = e_initial.
        # de_final = e_start - e_final.
        # e_final = e_start - de_final.
        # e_start = e_initial + de_initial.
        # e_final = (e_initial + de_initial) - pred_de_final

        e_final_guess = (e_initial + de_initial) - pred_de_final

        # If e_final_guess is much worse than threshold, discard
        # "would not be evaluated on the basis of this guess"
        # means it wouldn't even be in the top candidates.
        # We also need a lower limit for the threshold.
        # If we have no threshold (inf), don't discard.
        if energy_threshold == float('inf'):
            return False

        if e_final_guess > energy_threshold + 1.0:  # Tightened buffer
            self.stats["skipped"] += 1
            return True
        return False

    def report(self):
        if not self.active: return
        print(f"\n--- Smart Optimizer Stats ---")
        print(f"Total processed: {self.stats['total']}")
        print(f"Early skips:     {self.stats['skipped']}")
        if self.model:
            r2 = self.model.score(np.array([[s[0], s[1]] for s in self.samples]),
                                  np.array([s[2] for s in self.samples]))
            print(f"Model R^2:       {r2:.4f} (Samples: {len(self.samples)})")
