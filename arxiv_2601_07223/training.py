"""Training utilities for CUDA-Q variational classifiers."""

from __future__ import annotations

import math
import random
from typing import Callable, Iterable, List, Sequence, Tuple

from .classifier import ParityClassifier, mse_loss


class VariationalTrainer:
    """Single-parameter gradient descent with optional mini-batching."""

    def __init__(
        self,
        classifier: ParityClassifier,
        dataset: Iterable[Tuple[Tuple[int, int], int]],
        learning_rate: float,
        batch_size: int | None = None,
        seed: int | None = None,
    ):
        self.classifier = classifier
        self.dataset: List[Tuple[Tuple[int, int], int]] = list(dataset)
        if not self.dataset:
            raise ValueError("dataset must contain at least one sample")
        self.lr = learning_rate
        self.batch_size = batch_size
        self.rng = random.Random(seed)

    def _select_batch(self) -> Sequence[Tuple[Tuple[int, int], int]]:
        if not self.batch_size or self.batch_size >= len(self.dataset):
            return self.dataset
        return self.rng.sample(self.dataset, self.batch_size)

    def loss(self, theta: float, batch: Sequence[Tuple[Tuple[int, int], int]] | None = None):
        batch = batch or self.dataset
        total_loss = 0.0
        accuracy_hits = 0
        total_events = 0
        for bits, label in batch:
            stats = self.classifier(theta, bits)
            total_loss += mse_loss(stats.expectation, label)
            prediction = 0 if stats.expectation >= 0 else 1
            if prediction == label:
                accuracy_hits += 1
            total_events += 1
        return total_loss / len(batch), accuracy_hits / total_events if total_events else 0.0

    def gradient(self, theta: float, batch, shift: float = math.pi / 2) -> float:
        plus_loss, _ = self.loss(theta + shift, batch=batch)
        minus_loss, _ = self.loss(theta - shift, batch=batch)
        return 0.5 * (plus_loss - minus_loss)

    def train(
        self,
        theta: float,
        steps: int,
        callback: Callable[[int, float, float, float, float, int], None] | None = None,
    ):
        for step in range(steps):
            batch = self._select_batch()
            loss_value, accuracy = self.loss(theta, batch=batch)
            grad_value = self.gradient(theta, batch=batch)
            theta -= self.lr * grad_value
            print(
                f"[step {step:03d}] theta={theta:.4f} loss={loss_value:.4f} "
                f"grad={grad_value:.4f} acc={accuracy:.2%} batch={len(batch)}"
            )
            if callback:
                callback(step, theta, loss_value, grad_value, accuracy, len(batch))
        return theta
