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
    parser.add_argument(
        "--mode", 
        choices=["bare", "encoded", "encoded_logical"], 
        default="encoded", 
        help="Circuit variant: bare (unencoded), encoded (simplified rotations), encoded_logical (paper-correct with ancilla rotations)."
    )
    parser.add_argument("--shots", type=int, default=2048, help="Shots per training example.")
    parser.add_argument("--steps", type=int, default=80, help="Gradient descent steps.")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate.")
    parser.add_argument("--theta0", type=float, default=0.3, help="Initial parameter.")
    parser.add_argument("--gate-noise", type=float, default=0.0, help="Single-qubit depolarizing probability.")
    parser.add_argument(
        "--ancilla-noise",
        type=float,
        default=None,
        help="Ancilla qubit depolarizing probability (if different from gate-noise). Paper threshold: ~0.003-0.004.",
    )
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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate training curve plot after completion (requires matplotlib).",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Output path for plot (default: <log-json>.png). Only used if --plot is set.",
    )
    parser.add_argument(
        "--plot-title",
        type=str,
        default=None,
        help="Custom title for the plot.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cudaq.set_target(args.target)
    noise_model = build_noise_model(
        args.gate_noise, 
        args.two_qubit_noise_scale,
        ancilla_error=args.ancilla_noise,
    )
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
        
        # Generate plot if requested
        if args.plot:
            _generate_plot(args, log_path, log_entries)


def _generate_plot(args, log_path: Path, entries: List[dict]):
    """Generate training curve plot from logged data."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Install with: pip install matplotlib")
        return
    
    if not entries:
        print("Warning: No entries to plot.")
        return
    
    steps = [entry["step"] for entry in entries]
    accuracies = [entry["accuracy"] * 100.0 for entry in entries]  # Convert to percentage
    losses = [entry["loss"] for entry in entries]
    
    # Determine output path
    if args.plot_output:
        output_path = Path(args.plot_output)
    else:
        output_path = log_path.with_suffix(".png")
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # Accuracy plot
    ax1.plot(steps, accuracies, linewidth=2, color="#2E86AB", label="Accuracy")
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right")
    
    # Loss plot
    ax2.plot(steps, losses, linewidth=2, color="#A23B72", label="Loss")
    ax2.set_xlabel("Training Step", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right")
    
    # Title
    if args.plot_title:
        title = args.plot_title
    else:
        mode = args.mode.capitalize()
        dataset = args.dataset.upper()
        noise = args.gate_noise
        title = f"{dataset} - {mode} (noise={noise})"
    
    fig.suptitle(title, fontsize=13, fontweight="bold")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
