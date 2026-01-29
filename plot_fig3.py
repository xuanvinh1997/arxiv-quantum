"""Plot training curves to compare with Figure 3 from arXiv:2601.07223."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - plotting dependency guard
    raise SystemExit(
        "matplotlib is required to run this script. Install it via pip install matplotlib."
    ) from exc


def _load_log(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    entries = data.get("entries") or []
    if not entries:
        raise ValueError(f"Log file {path} has no entries.")
    steps = [int(entry["step"]) for entry in entries]
    accuracies = [float(entry["accuracy"]) * 100.0 for entry in entries]  # percent
    metadata: Dict[str, object] = data.get("args", {})
    metadata["log_path"] = str(path)
    metadata["dataset_size"] = data.get("dataset_size")
    metadata["label_counts"] = data.get("label_counts")
    return {
        "path": path,
        "steps": steps,
        "accuracies": accuracies,
        "metadata": metadata,
        "entries": entries,
    }


def _label_from_metadata(meta: Dict[str, object]) -> str:
    dataset = meta.get("dataset", "unknown")
    mode = meta.get("mode", "run")
    label = f"{mode.capitalize()}"
    if dataset == "mnist":
        pos = meta.get("mnist_digit_positive")
        neg = meta.get("mnist_digit_negative")
        label += f" ({pos} vs {neg})"
    shots = meta.get("shots")
    if shots is not None:
        label += f", shots={shots}"
    noise = meta.get("gate_noise")
    if noise:
        label += f", gate_noise={noise}"
    return label


def _noise_series_label(meta: Dict[str, object]) -> str:
    dataset = meta.get("dataset", "dataset")
    mode = str(meta.get("mode", "run")).capitalize()
    if dataset == "mnist":
        pos = meta.get("mnist_digit_positive")
        neg = meta.get("mnist_digit_negative")
        dataset_label = f"MNIST {pos} vs {neg}"
    else:
        dataset_label = str(dataset).upper()
    return f"{dataset_label} - {mode}"


def _select_accuracy(entries: List[Dict[str, object]], metric: str, mean_window: int) -> float:
    values = [float(entry["accuracy"]) * 100.0 for entry in entries]
    if metric == "last":
        return values[-1]
    if metric == "best":
        return max(values)
    if metric == "mean":
        window = max(1, min(mean_window, len(values)))
        tail = values[-window:]
        return sum(tail) / len(tail)
    raise ValueError(f"Unknown noise metric '{metric}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CUDA-Q training logs to reproduce Figure 3 trends."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="Path(s) to JSON files produced with --log-json in arxiv_2601_07223.cli.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fig3_reproduction.png",
        help="Output image path (default: fig3_reproduction.png).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="MNIST accuracy (Figure 3 reproduction)",
        help="Custom plot title.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=0.0,
        help="Lower bound for accuracy axis (default: 0).",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=100.0,
        help="Upper bound for accuracy axis (default: 100).",
    )
    parser.add_argument(
        "--plot-noise",
        action="store_true",
        help="Plot final accuracy vs. gate noise (paper's noise sweep) instead of accuracy vs. training step.",
    )
    parser.add_argument(
        "--noise-metric",
        choices=["last", "best", "mean"],
        default="best",
        help="Statistic applied to each run when --plot-noise is enabled.",
    )
    parser.add_argument(
        "--noise-mean-window",
        type=int,
        default=5,
        help="Window size for the 'mean' noise metric (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig, ax = plt.subplots(figsize=(6, 4))
    records = [_load_log(Path(p)) for p in args.logs]
    if args.plot_noise:
        _plot_noise_sweep(
            ax,
            records,
            metric=args.noise_metric,
            mean_window=args.noise_mean_window,
        )
        ax.set_xlabel("Single-qubit depolarizing probability")
        ax.set_ylabel("Accuracy (%)")
    else:
        _plot_training_steps(ax, records)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_title(args.title)
    ax.grid(alpha=0.3)
    ax.legend()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")


def _plot_training_steps(ax, records: List[Dict[str, object]]) -> None:
    for record in records:
        steps = record["steps"]
        accuracies = record["accuracies"]
        meta = record["metadata"]
        label = _label_from_metadata(meta)
        ax.plot(steps, accuracies, label=label, linewidth=2)


def _plot_noise_sweep(
    ax,
    records: List[Dict[str, object]],
    *,
    metric: str,
    mean_window: int,
) -> None:
    series: Dict[str, Dict[float, List[float]]] = {}
    for record in records:
        meta = record["metadata"]
        entries = record["entries"]
        noise = float(meta.get("gate_noise") or 0.0)
        series_label = _noise_series_label(meta)
        accuracy = _select_accuracy(entries, metric, mean_window)
        series.setdefault(series_label, {}).setdefault(noise, []).append(accuracy)

    if not series:
        raise ValueError("No logs available to plot.")

    for label, noise_map in series.items():
        sorted_items = sorted(noise_map.items(), key=lambda item: item[0])
        noises = [noise for noise, _ in sorted_items]
        accuracies = [sum(vals) / len(vals) for _, vals in sorted_items]
        ax.plot(noises, accuracies, label=label, marker="o", linewidth=2)


if __name__ == "__main__":
    main()
