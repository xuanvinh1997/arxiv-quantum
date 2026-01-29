"""Command-line interface for CUDA-Q training experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import cudaq

from .classifier import ParityClassifier, build_noise_model
from .data import mnist_binary_dataset, parity_dataset
from .training import VariationalTrainer


def _load_dataset(args) -> Iterable[Tuple[Tuple[int, int], int]]:
    if args.dataset == "parity":
        dataset = parity_dataset()
        print("Loaded parity dataset with 4 samples.")
        return dataset

    dataset = mnist_binary_dataset(
        digit_positive=args.mnist_digit_positive,
        digit_negative=args.mnist_digit_negative,
        limit_per_class=args.mnist_limit,
        data_dir=args.mnist_data_dir,
        threshold=args.mnist_threshold,
        train_split=not args.mnist_use_test,
    )
    counts = {
        args.mnist_digit_positive: sum(1 for _, label in dataset if label == 0),
        args.mnist_digit_negative: sum(1 for _, label in dataset if label == 1),
    }
    print(
        "Loaded MNIST dataset with counts "
        + json.dumps(counts)
        + f" (total {len(dataset)} samples)."
    )
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA-Q parity classifier inspired by arXiv:2601.07223.")
    parser.add_argument("--dataset", choices=["parity", "mnist"], default="parity", help="Training dataset.")
    parser.add_argument("--mode", choices=["bare", "encoded"], default="encoded", help="Circuit variant.")
    parser.add_argument("--shots", type=int, default=2048, help="Shots per training example.")
    parser.add_argument("--steps", type=int, default=80, help="Gradient descent steps.")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate.")
    parser.add_argument("--theta0", type=float, default=0.3, help="Initial parameter.")
    parser.add_argument("--gate-noise", type=float, default=0.0, help="Single-qubit depolarizing probability.")
    parser.add_argument(
        "--two-qubit-noise-scale",
        type=float,
        default=2.0,
        help="Multiplier applied to two-qubit gate noise probability.",
    )
    parser.add_argument(
        "--syndrome-rounds",
        type=int,
        default=1,
        help="Number of stabilizer extraction stages for the encoded mode.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="density-matrix-cpu",
        help="CUDA-Q target backend (density-matrix-cpu recommended for noise).",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size. Defaults to full batch.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for batching.")

    # MNIST-specific arguments
    parser.add_argument("--mnist-digit-positive", type=int, default=0, help="Label mapped to class 0.")
    parser.add_argument("--mnist-digit-negative", type=int, default=1, help="Label mapped to class 1.")
    parser.add_argument("--mnist-limit", type=int, default=256, help="Samples per digit to keep.")
    parser.add_argument("--mnist-data-dir", type=str, default="./data", help="Where MNIST is cached/downloaded.")
    parser.add_argument(
        "--mnist-threshold",
        type=float,
        default=None,
        help="Absolute threshold for binarizing quadrant averages (default: relative).",
    )
    parser.add_argument(
        "--mnist-use-test",
        action="store_true",
        help="Use the MNIST test split instead of the training split.",
    )
    parser.add_argument(
        "--log-json",
        type=str,
        default=None,
        help="Optional path to store per-step metrics for plotting (Figure 3 reproduction).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cudaq.set_target(args.target)
    noise_model = build_noise_model(args.gate_noise, args.two_qubit_noise_scale)
    dataset = list(_load_dataset(args))
    classifier = ParityClassifier(
        mode=args.mode,
        shots=args.shots,
        syndrome_rounds=args.syndrome_rounds,
        noise_model=noise_model,
    )
    trainer = VariationalTrainer(
        classifier=classifier,
        dataset=dataset,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    log_entries: List[dict] = []

    def _log_callback(step: int, theta: float, loss: float, grad: float, acc: float, batch_size: int):
        log_entries.append(
            {
                "step": step,
                "theta": theta,
                "loss": loss,
                "gradient": grad,
                "accuracy": acc,
                "batch_size": batch_size,
            }
        )

    callback = _log_callback if args.log_json else None
    final_theta = trainer.train(theta=args.theta0, steps=args.steps, callback=callback)
    print(f"Optimized theta: {final_theta:.4f}")

    if args.log_json:
        log_path = Path(args.log_json)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        label_counts: dict[int, int] = {}
        for _, label in dataset:
            label_counts[label] = label_counts.get(label, 0) + 1
        payload = {
            "args": vars(args),
            "dataset_size": len(dataset),
            "label_counts": label_counts,
            "entries": log_entries,
        }
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote training log to {log_path}")


if __name__ == "__main__":
    main()
