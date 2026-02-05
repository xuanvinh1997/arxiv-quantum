#!/usr/bin/env python3
"""
Experiment runner for arXiv:2601.07223 paper reproduction.

This script runs all experiments described in the paper:
1. Zero-noise baseline validation
2. Full noise sweep (Figure 4 reproduction)
3. Syndrome round analysis (Section 4.3)
4. Ancilla noise threshold discovery (Section 4.3.1-2)
5. MNIST binary classification experiments

Usage:
    python run_experiments.py --experiment all
    python run_experiments.py --experiment noise-sweep
    python run_experiments.py --experiment baseline --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    dataset: str
    mode: str
    gate_noise: float
    shots: int = 2048
    steps: int = 100
    num_layers: int = 1
    syndrome_rounds: int = 1
    ancilla_noise: Optional[float] = None
    two_qubit_noise_scale: float = 2.0
    env_noise_interval: Optional[int] = None
    batch_size: Optional[int] = None
    # MNIST specific
    mnist_digit_positive: int = 0
    mnist_digit_negative: int = 1
    mnist_limit: int = 256
    
    def to_command(self, output_dir: Path) -> List[str]:
        """Convert config to CLI command."""
        log_file = output_dir / f"{self.name}.json"
        cmd = [
            sys.executable, "-m", "arxiv_2601_07223.cli",
            "--dataset", self.dataset,
            "--mode", self.mode,
            "--gate-noise", str(self.gate_noise),
            "--shots", str(self.shots),
            "--steps", str(self.steps),
            "--num-layers", str(self.num_layers),
            "--syndrome-rounds", str(self.syndrome_rounds),
            "--two-qubit-noise-scale", str(self.two_qubit_noise_scale),
            "--log-json", str(log_file),
            "--plot",
        ]
        if self.ancilla_noise is not None:
            cmd.extend(["--ancilla-noise", str(self.ancilla_noise)])
        if self.env_noise_interval is not None:
            cmd.extend(["--env-noise-interval", str(self.env_noise_interval)])
        if self.batch_size is not None:
            cmd.extend(["--batch-size", str(self.batch_size)])
        if self.dataset == "mnist":
            cmd.extend([
                "--mnist-digit-positive", str(self.mnist_digit_positive),
                "--mnist-digit-negative", str(self.mnist_digit_negative),
                "--mnist-limit", str(self.mnist_limit),
            ])
        return cmd


def get_baseline_experiments() -> List[ExperimentConfig]:
    """Phase 1: Zero-noise baseline validation."""
    return [
        ExperimentConfig(
            name="baseline_parity_bare_no_noise",
            dataset="parity",
            mode="bare",
            gate_noise=0.0,
            steps=100,
        ),
        ExperimentConfig(
            name="baseline_parity_encoded_no_noise",
            dataset="parity",
            mode="encoded",
            gate_noise=0.0,
            steps=100,
        ),
        ExperimentConfig(
            name="baseline_parity_encoded_logical_no_noise",
            dataset="parity",
            mode="encoded_logical",
            gate_noise=0.0,
            steps=100,
        ),
    ]


def get_noise_sweep_experiments() -> List[ExperimentConfig]:
    """Phase 2: Full noise sweep (Figure 4 reproduction)."""
    noise_levels = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
    experiments = []
    
    for p in noise_levels:
        # Bare mode
        experiments.append(ExperimentConfig(
            name=f"noise_sweep_bare_p_{p:.3f}".replace(".", "_"),
            dataset="parity",
            mode="bare",
            gate_noise=p,
            shots=4096,
            steps=100,
        ))
        # Encoded mode with syndrome extraction
        experiments.append(ExperimentConfig(
            name=f"noise_sweep_encoded_p_{p:.3f}".replace(".", "_"),
            dataset="parity",
            mode="encoded",
            gate_noise=p,
            shots=4096,
            steps=100,
            syndrome_rounds=2,
        ))
    
    return experiments


def get_syndrome_round_experiments() -> List[ExperimentConfig]:
    """Phase 3: Syndrome round analysis at fixed noise."""
    syndrome_rounds = [0, 1, 2, 3, 4, 5]
    experiments = []
    
    for rounds in syndrome_rounds:
        experiments.append(ExperimentConfig(
            name=f"syndrome_rounds_{rounds}_p_0_005",
            dataset="parity",
            mode="encoded",
            gate_noise=0.005,
            shots=4096,
            steps=100,
            syndrome_rounds=rounds,
        ))
    
    return experiments


def get_ancilla_threshold_experiments() -> List[ExperimentConfig]:
    """Phase 4: Ancilla noise threshold discovery (Section 4.3.1-2)."""
    ancilla_noise_levels = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
    experiments = []
    
    for anc_p in ancilla_noise_levels:
        experiments.append(ExperimentConfig(
            name=f"ancilla_threshold_anc_{anc_p:.3f}".replace(".", "_"),
            dataset="parity",
            mode="encoded_logical",
            gate_noise=0.003,
            ancilla_noise=anc_p,
            shots=4096,
            steps=100,
            syndrome_rounds=2,
        ))
    
    return experiments


def get_environmental_noise_experiments() -> List[ExperimentConfig]:
    """Environmental noise model experiments (every N gates)."""
    noise_levels = [0.001, 0.003, 0.005]
    experiments = []
    
    for p in noise_levels:
        # With environmental noise (every 4 gates, as in paper)
        experiments.append(ExperimentConfig(
            name=f"env_noise_p_{p:.3f}_interval_4".replace(".", "_"),
            dataset="parity",
            mode="encoded",
            gate_noise=p,
            env_noise_interval=4,
            shots=4096,
            steps=100,
            syndrome_rounds=2,
        ))
        # Without environmental noise (gate noise only) for comparison
        experiments.append(ExperimentConfig(
            name=f"gate_noise_only_p_{p:.3f}".replace(".", "_"),
            dataset="parity",
            mode="encoded",
            gate_noise=p,
            env_noise_interval=None,
            shots=4096,
            steps=100,
            syndrome_rounds=2,
        ))
    
    return experiments


def get_deep_circuit_experiments() -> List[ExperimentConfig]:
    """Deep circuit experiments (multiple layers)."""
    layer_counts = [1, 2, 5, 10, 25, 50]
    experiments = []
    
    for num_layers in layer_counts:
        # No noise - test expressibility
        experiments.append(ExperimentConfig(
            name=f"deep_circuit_bare_layers_{num_layers}_no_noise",
            dataset="parity",
            mode="bare",
            gate_noise=0.0,
            num_layers=num_layers,
            shots=2048,
            steps=100,
        ))
        # With noise
        experiments.append(ExperimentConfig(
            name=f"deep_circuit_bare_layers_{num_layers}_p_0_003",
            dataset="parity",
            mode="bare",
            gate_noise=0.003,
            num_layers=num_layers,
            shots=2048,
            steps=100,
        ))
        experiments.append(ExperimentConfig(
            name=f"deep_circuit_encoded_layers_{num_layers}_p_0_003",
            dataset="parity",
            mode="encoded",
            gate_noise=0.003,
            num_layers=num_layers,
            shots=2048,
            steps=100,
            syndrome_rounds=2,
        ))
    
    return experiments


def get_mnist_experiments() -> List[ExperimentConfig]:
    """Phase 5: MNIST binary classification experiments."""
    digit_pairs = [(0, 1), (3, 8), (4, 9), (6, 8)]
    noise_levels = [0.0, 0.003, 0.005]
    experiments = []
    
    for pos, neg in digit_pairs:
        for p in noise_levels:
            # Bare mode
            experiments.append(ExperimentConfig(
                name=f"mnist_{pos}v{neg}_bare_p_{p:.3f}".replace(".", "_"),
                dataset="mnist",
                mode="bare",
                gate_noise=p,
                shots=2048,
                steps=150,
                batch_size=32,
                mnist_digit_positive=pos,
                mnist_digit_negative=neg,
                mnist_limit=256,
            ))
            # Encoded mode
            if p > 0:  # Skip encoded for zero noise (same as bare)
                experiments.append(ExperimentConfig(
                    name=f"mnist_{pos}v{neg}_encoded_p_{p:.3f}".replace(".", "_"),
                    dataset="mnist",
                    mode="encoded",
                    gate_noise=p,
                    shots=2048,
                    steps=150,
                    batch_size=32,
                    syndrome_rounds=2,
                    mnist_digit_positive=pos,
                    mnist_digit_negative=neg,
                    mnist_limit=256,
                ))
    
    return experiments


def get_all_experiments() -> List[ExperimentConfig]:
    """Get all experiments for full paper reproduction."""
    return (
        get_baseline_experiments() +
        get_noise_sweep_experiments() +
        get_syndrome_round_experiments() +
        get_ancilla_threshold_experiments() +
        get_environmental_noise_experiments() +
        get_deep_circuit_experiments() +
        get_mnist_experiments()
    )


EXPERIMENT_GROUPS = {
    "baseline": get_baseline_experiments,
    "noise-sweep": get_noise_sweep_experiments,
    "syndrome-rounds": get_syndrome_round_experiments,
    "ancilla-threshold": get_ancilla_threshold_experiments,
    "environmental": get_environmental_noise_experiments,
    "deep-circuits": get_deep_circuit_experiments,
    "mnist": get_mnist_experiments,
    "all": get_all_experiments,
}


def run_experiments(
    experiments: List[ExperimentConfig],
    output_dir: Path,
    dry_run: bool = False,
    continue_on_error: bool = True,
) -> dict:
    """Run a list of experiments and collect results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "start_time": datetime.now().isoformat(),
        "experiments": [],
        "summary": {"total": len(experiments), "success": 0, "failed": 0, "skipped": 0},
    }
    
    for i, config in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(experiments)}] Running: {config.name}")
        print(f"{'='*60}")
        
        cmd = config.to_command(output_dir)
        print(f"Command: {' '.join(cmd)}")
        
        if dry_run:
            print("(dry run - skipping)")
            results["experiments"].append({
                "name": config.name,
                "status": "skipped",
                "command": cmd,
            })
            results["summary"]["skipped"] += 1
            continue
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout per experiment
            )
            
            if result.returncode == 0:
                print(f"✓ Success")
                results["experiments"].append({
                    "name": config.name,
                    "status": "success",
                    "command": cmd,
                    "stdout": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
                })
                results["summary"]["success"] += 1
            else:
                print(f"✗ Failed (exit code {result.returncode})")
                print(f"stderr: {result.stderr[-500:]}")
                results["experiments"].append({
                    "name": config.name,
                    "status": "failed",
                    "command": cmd,
                    "returncode": result.returncode,
                    "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
                })
                results["summary"]["failed"] += 1
                if not continue_on_error:
                    break
                    
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout (1 hour)")
            results["experiments"].append({
                "name": config.name,
                "status": "timeout",
                "command": cmd,
            })
            results["summary"]["failed"] += 1
            if not continue_on_error:
                break
        except Exception as e:
            print(f"✗ Error: {e}")
            results["experiments"].append({
                "name": config.name,
                "status": "error",
                "command": cmd,
                "error": str(e),
            })
            results["summary"]["failed"] += 1
            if not continue_on_error:
                break
    
    results["end_time"] = datetime.now().isoformat()
    
    # Save results summary
    summary_file = output_dir / "experiment_summary.json"
    with summary_file.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults summary saved to {summary_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run paper experiments for arXiv:2601.07223",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment groups:
  baseline          Zero-noise validation (bare, encoded, encoded_logical)
  noise-sweep       Full noise sweep p=0.001-0.010 (Figure 4)
  syndrome-rounds   Syndrome round analysis 0-5 rounds (Section 4.3)
  ancilla-threshold Ancilla noise threshold discovery (Section 4.3.1-2)
  environmental     Environmental noise model comparison
  deep-circuits     Multi-layer circuit experiments (1-50 layers)
  mnist             MNIST binary classification (0v1, 3v8, 4v9, 6v8)
  all               Run all experiments
        """
    )
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENT_GROUPS.keys()),
        default="baseline",
        help="Which experiment group to run (default: baseline)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments",
        help="Output directory for results (default: results/experiments)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop on first error instead of continuing",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiments in the group without running",
    )
    
    args = parser.parse_args()
    
    experiments = EXPERIMENT_GROUPS[args.experiment]()
    output_dir = Path(args.output_dir) / args.experiment
    
    if args.list:
        print(f"Experiments in '{args.experiment}' ({len(experiments)} total):\n")
        for i, exp in enumerate(experiments, 1):
            print(f"  {i:3d}. {exp.name}")
            print(f"       mode={exp.mode}, noise={exp.gate_noise}, layers={exp.num_layers}")
        return
    
    print(f"Running {len(experiments)} experiments from '{args.experiment}' group")
    print(f"Output directory: {output_dir}")
    
    results = run_experiments(
        experiments,
        output_dir,
        dry_run=args.dry_run,
        continue_on_error=not args.stop_on_error,
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total:   {results['summary']['total']}")
    print(f"Success: {results['summary']['success']}")
    print(f"Failed:  {results['summary']['failed']}")
    print(f"Skipped: {results['summary']['skipped']}")


if __name__ == "__main__":
    main()
