"""
SEGQE Paper Figure Reproduction (arXiv:2602.17615)
Generates publication-quality plots matching the paper's main figures.

Usage:
    python segqe_plots.py --figure all              # all figures (slow)
    python segqe_plots.py --figure 1 --max-n 6      # quick test for Figure 1
    python segqe_plots.py --figure 3 --n-instances 5 --max-n 5  # quick test
    python segqe_plots.py --figure 4 --n-instances 5             # quick test
    python segqe_plots.py --recompute               # ignore cache
"""

import sys
import json
import argparse
import time
import io
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Import from existing PennyLane implementation
sys.path.insert(0, str(Path(__file__).parent))
from segqe_pennylane import (
    segqe,
    transverse_field_ising,
    random_local_hamiltonian,
    hamiltonian_to_pauli_terms,
    exact_ground_state_energy,
)

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

THRESHOLD = 1e-3
CACHE_DIR = Path(__file__).parent / "plot_cache"
FIGURES_DIR = Path(__file__).parent / "figures"


# ──────────────────────────────────────────────────────────
# Plot Style
# ──────────────────────────────────────────────────────────

def setup_plot_style():
    """Configure matplotlib for paper-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'lines.markersize': 5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'xtick.top': True,
        'ytick.right': True,
    })

    # Try LaTeX rendering
    try:
        plt.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        })
        fig_test = plt.figure()
        fig_test.text(0.5, 0.5, r'$\delta E$')
        fig_test.savefig(io.BytesIO(), format='png')
        plt.close(fig_test)
    except Exception:
        plt.rcParams['text.usetex'] = False
        print("LaTeX not available, using mathtext renderer")


# ──────────────────────────────────────────────────────────
# Cache Layer
# ──────────────────────────────────────────────────────────

def save_cache(name: str, data: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Cached: {path}")


def load_cache(name: str) -> dict | None:
    path = CACHE_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"  Loaded cache: {path}")
        return data
    return None


# ──────────────────────────────────────────────────────────
# Data Generation: Figure 1 — TFI Model
# ──────────────────────────────────────────────────────────

def generate_tfi_data(
    system_sizes: list[int],
    recompute: bool = False,
) -> dict:
    """Run SEGQE on TFI for critical (J=w=1) and gapped (J=w/2) regimes."""
    cache_name = f"tfi_n{min(system_sizes)}-{max(system_sizes)}"
    if not recompute:
        cached = load_cache(cache_name)
        if cached is not None:
            return cached

    data = {
        "critical": {"n": [], "rel_error": [], "infidelity": [], "n_gates": []},
        "gapped":   {"n": [], "rel_error": [], "infidelity": [], "n_gates": []},
    }

    for regime, J_val in [("critical", 1.0), ("gapped", 0.5)]:
        for n in system_sizes:
            t0 = time.time()
            print(f"  TFI {regime}: n={n} ...", end=" ", flush=True)

            H = transverse_field_ising(n, w=1.0, J=J_val, periodic=False)
            ham_terms = hamiltonian_to_pauli_terms(H)
            E_exact, gs_vec = exact_ground_state_energy(ham_terms, n)

            result = segqe(
                hamiltonian=H,
                num_qubits=n,
                max_depth=max(4 * n, 50),
                threshold=THRESHOLD,
                locality=2,
                nearest_neighbor_only=False,
                use_exact=True,
                seed=42,
                verbose=False,
            )

            rel_error = abs((E_exact - result.energy) / E_exact)
            fidelity = abs(np.vdot(gs_vec, result.final_state)) ** 2
            infidelity = 1.0 - fidelity

            data[regime]["n"].append(n)
            data[regime]["rel_error"].append(float(rel_error))
            data[regime]["infidelity"].append(float(max(infidelity, 1e-15)))
            data[regime]["n_gates"].append(result.num_iterations)

            dt = time.time() - t0
            print(f"E={result.energy:.6f}, δE={rel_error:.2e}, "
                  f"1-F={infidelity:.2e}, gates={result.num_iterations} ({dt:.1f}s)")

    save_cache(cache_name, data)
    return data


# ──────────────────────────────────────────────────────────
# Data Generation: Figure 3 — Random Local Hamiltonians
# ──────────────────────────────────────────────────────────

def generate_random_data(
    system_sizes: list[int],
    num_instances: int = 50,
    recompute: bool = False,
) -> dict:
    """Run SEGQE on random Hamiltonians with P² and NN-P² gate sets."""
    cache_name = f"random_n{min(system_sizes)}-{max(system_sizes)}_inst{num_instances}"
    if not recompute:
        cached = load_cache(cache_name)
        if cached is not None:
            return cached

    data = {}
    for gs_name, nn_only in [("NN_P2", True), ("P2", False)]:
        data[gs_name] = {"n": [], "rel_error": [], "fidelity": [], "n_gates": []}

        for n in system_sizes:
            t0 = time.time()
            rng_ham = np.random.default_rng(12345)
            errors, fids, gates = [], [], []

            for inst in range(num_instances):
                H = random_local_hamiltonian(n, rng=rng_ham)
                ham_terms = hamiltonian_to_pauli_terms(H)
                E_exact, gs_vec = exact_ground_state_energy(ham_terms, n)

                result = segqe(
                    hamiltonian=H,
                    num_qubits=n,
                    max_depth=max(4 * n, 50),
                    threshold=THRESHOLD,
                    locality=2,
                    nearest_neighbor_only=nn_only,
                    use_exact=True,
                    seed=42 + inst,
                    verbose=False,
                )

                rel_err = abs((E_exact - result.energy) / E_exact)
                fid = abs(np.vdot(gs_vec, result.final_state)) ** 2

                errors.append(float(rel_err))
                fids.append(float(fid))
                gates.append(result.num_iterations)

            data[gs_name]["n"].append(n)
            data[gs_name]["rel_error"].append(errors)
            data[gs_name]["fidelity"].append(fids)
            data[gs_name]["n_gates"].append(gates)

            dt = time.time() - t0
            print(f"  Random {gs_name}: n={n}, "
                  f"mean_δE={np.mean(errors):.4f}, mean_F={np.mean(fids):.4f}, "
                  f"mean_gates={np.mean(gates):.1f} ({dt:.1f}s)")

    save_cache(cache_name, data)
    return data


# ──────────────────────────────────────────────────────────
# Data Generation: Figure 4 — Gate Set Dependence
# ──────────────────────────────────────────────────────────

def generate_gateset_data(
    n: int = 5,
    num_instances: int = 50,
    max_gates: int = 80,
    recompute: bool = False,
) -> dict:
    """Run SEGQE with P² and NN-P² gate sets at fixed n, tracking full energy histories."""
    cache_name = f"gateset_n{n}_inst{num_instances}"
    if not recompute:
        cached = load_cache(cache_name)
        if cached is not None:
            return cached

    rng_ham = np.random.default_rng(99999)

    # Pre-generate Hamiltonians
    hamiltonians_data = []
    for inst in range(num_instances):
        H = random_local_hamiltonian(n, rng=rng_ham)
        ham_terms = hamiltonian_to_pauli_terms(H)
        E_exact, gs_vec = exact_ground_state_energy(ham_terms, n)
        hamiltonians_data.append((H, E_exact, gs_vec))

    data = {}
    for gs_name, nn_only in [("NN_P2", True), ("P2", False)]:
        t0 = time.time()
        histories = []
        exact_energies = []
        n_gates_final = []

        for inst in range(num_instances):
            H, E_exact, gs_vec = hamiltonians_data[inst]

            result = segqe(
                hamiltonian=H,
                num_qubits=n,
                max_depth=max_gates,
                threshold=THRESHOLD,
                locality=2,
                nearest_neighbor_only=nn_only,
                use_exact=True,
                seed=42 + inst,
                verbose=False,
            )

            histories.append([float(e) for e in result.energy_history])
            exact_energies.append(float(E_exact))
            n_gates_final.append(result.num_iterations)

        data[gs_name] = {
            "energy_histories": histories,
            "exact_energies": exact_energies,
            "n_gates_final": n_gates_final,
        }
        dt = time.time() - t0
        print(f"  Gateset {gs_name}: mean_gates={np.mean(n_gates_final):.1f} ({dt:.1f}s)")

    save_cache(cache_name, data)
    return data


# ──────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────

def compute_stats(values_per_n: list[list[float]]) -> dict:
    """Compute mean, median, and 2σ CI of the mean for each system size."""
    means, medians, ci_low, ci_high = [], [], [], []
    for vals in values_per_n:
        arr = np.array(vals)
        m = np.mean(arr)
        med = np.median(arr)
        se = np.std(arr, ddof=1) / np.sqrt(len(arr))
        means.append(m)
        medians.append(med)
        ci_low.append(m - 2 * se)
        ci_high.append(m + 2 * se)
    return {
        'mean': np.array(means),
        'median': np.array(medians),
        'ci_low': np.array(ci_low),
        'ci_high': np.array(ci_high),
    }


# ──────────────────────────────────────────────────────────
# Plot: Figure 1 — TFI Convergence
# ──────────────────────────────────────────────────────────

def plot_figure1(data: dict, save_path: Path = None) -> plt.Figure:
    """
    TFI convergence: 3 subplots.
    Top:    δE (log) vs n
    Middle: 1-F (log) vs n
    Bottom: N_gates (linear) vs n
    """
    fig, axes = plt.subplots(3, 1, figsize=(3.6, 5.5), sharex=True)

    colors = {'critical': 'tab:blue', 'gapped': 'tab:orange'}
    labels = {'critical': r'$J = w = 1$', 'gapped': r'$J = w/2$'}

    for regime in ['critical', 'gapped']:
        d = data[regime]
        n_vals = d['n']
        c = colors[regime]
        lbl = labels[regime]

        axes[0].semilogy(n_vals, d['rel_error'], 'o-', color=c, label=lbl, markersize=5)
        axes[1].semilogy(n_vals, d['infidelity'], 'o-', color=c, label=lbl, markersize=5)
        axes[2].plot(n_vals, d['n_gates'], 'o-', color=c, label=lbl, markersize=5)

    axes[0].set_ylabel(r'$\delta E$')
    axes[1].set_ylabel(r'$1 - \mathcal{F}$')
    axes[2].set_ylabel(r'$N_{\mathrm{Gates}}$')
    axes[2].set_xlabel(r'$n$')

    axes[0].legend(loc='lower right', frameon=False)
    axes[2].set_xticks(data['critical']['n'])

    plt.subplots_adjust(hspace=0.08)

    if save_path:
        fig.savefig(save_path, format='pdf')
        print(f"Saved: {save_path}")
    return fig


# ──────────────────────────────────────────────────────────
# Plot: Figure 3 — Random Local Hamiltonians
# ──────────────────────────────────────────────────────────

def plot_figure3(data: dict, save_path: Path = None) -> plt.Figure:
    """
    Random Hamiltonians: 3 subplots (all linear scale).
    Top:    δE vs n
    Middle: F vs n
    Bottom: N_gates vs n
    Solid=mean, dashed=median, shaded=2σ CI
    """
    fig, axes = plt.subplots(3, 1, figsize=(3.6, 5.5), sharex=True)

    styles = {
        'NN_P2': {'color': 'tab:blue',  'label': r'NN-$\mathcal{P}^2$'},
        'P2':    {'color': 'tab:orange', 'label': r'$\mathcal{P}^2$'},
    }

    for gs_name in ['NN_P2', 'P2']:
        if gs_name not in data:
            continue
        d = data[gs_name]
        n_vals = np.array(d['n'])
        c = styles[gs_name]['color']
        lbl = styles[gs_name]['label']

        for ax_idx, key in enumerate(['rel_error', 'fidelity', 'n_gates']):
            stats = compute_stats(d[key])
            ax = axes[ax_idx]

            ax.plot(n_vals, stats['mean'], '-', color=c,
                    label=f'{lbl} (mean)', linewidth=1.2)
            ax.plot(n_vals, stats['median'], '--', color=c,
                    label=f'{lbl} (median)', linewidth=1.0, alpha=0.7)
            ax.fill_between(n_vals, stats['ci_low'], stats['ci_high'],
                            color=c, alpha=0.15)

    axes[0].set_ylabel(r'$\delta E$')
    axes[1].set_ylabel(r'$\mathcal{F}$')
    axes[2].set_ylabel(r'$N_{\mathrm{Gates}}$')
    axes[2].set_xlabel(r'$n$')

    # Compact legend in top panel
    axes[0].legend(loc='upper left', frameon=False, fontsize=7, ncol=1)
    axes[2].set_xticks(data[list(data.keys())[0]]['n'])

    plt.subplots_adjust(hspace=0.08)

    if save_path:
        fig.savefig(save_path, format='pdf')
        print(f"Saved: {save_path}")
    return fig


# ──────────────────────────────────────────────────────────
# Plot: Figure 4 — Gate Set Dependence
# ──────────────────────────────────────────────────────────

def plot_figure4(data: dict, save_path: Path = None) -> plt.Figure:
    """
    Gate set comparison at fixed n=5: δE (log) vs N_gates.
    Solid=mean, shaded=1σ CI.
    Horizontal solid lines = final mean error.
    Vertical dashed lines = mean convergence gate count.
    """
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.8))

    styles = {
        'NN_P2': {'color': 'tab:blue',  'label': r'NN-$\mathcal{P}^2$'},
        'P2':    {'color': 'tab:orange', 'label': r'$\mathcal{P}^2$'},
    }

    max_gate_count = 0

    for gs_name in ['NN_P2', 'P2']:
        if gs_name not in data:
            continue
        d = data[gs_name]
        c = styles[gs_name]['color']
        lbl = styles[gs_name]['label']

        histories = d['energy_histories']
        exact_es = d['exact_energies']
        n_gates_final = d['n_gates_final']

        max_len = max(len(h) for h in histories)
        max_gate_count = max(max_gate_count, max_len)

        # Compute relative error at each gate index across all instances
        # After convergence, hold the final energy
        rel_errors_by_gate = []
        for k in range(max_len):
            errs = []
            for inst_idx, h in enumerate(histories):
                e_k = h[min(k, len(h) - 1)]
                e_exact = exact_es[inst_idx]
                rel_err = abs((e_exact - e_k) / e_exact)
                errs.append(max(rel_err, 1e-15))
            rel_errors_by_gate.append(errs)

        gate_indices = np.arange(max_len)
        means = np.array([np.mean(e) for e in rel_errors_by_gate])
        stds = np.array([np.std(e, ddof=1) for e in rel_errors_by_gate])
        se = stds / np.sqrt(len(histories))

        # Mean curve + 1σ CI shading
        ax.semilogy(gate_indices, means, '-', color=c, label=lbl, linewidth=1.2)
        ax.fill_between(gate_indices,
                        np.maximum(means - se, 1e-15),
                        means + se,
                        color=c, alpha=0.15)

        # Final mean error (horizontal line)
        final_errors = []
        for inst_idx, h in enumerate(histories):
            e_exact = exact_es[inst_idx]
            final_errors.append(abs((e_exact - h[-1]) / e_exact))
        mean_final_error = np.mean(final_errors)
        ax.axhline(mean_final_error, color=c, linestyle='-', linewidth=0.6, alpha=0.5)

        # Mean convergence gate count (vertical dashed line)
        mean_final_gates = np.mean(n_gates_final)
        ax.axvline(mean_final_gates, color=c, linestyle='--', linewidth=0.8, alpha=0.6)

    ax.set_xlabel(r'$N_{\mathrm{Gates}}$')
    ax.set_ylabel(r'$\delta E$')
    ax.legend(loc='upper right', frameon=False)
    ax.set_xlim(0, min(max_gate_count + 2, 80))

    if save_path:
        fig.savefig(save_path, format='pdf')
        print(f"Saved: {save_path}")
    return fig


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate SEGQE paper figures (arXiv:2602.17615)"
    )
    parser.add_argument(
        '--figure', type=str, default='all',
        choices=['1', '3', '4', 'all'],
        help='Which figure to generate (default: all)',
    )
    parser.add_argument('--recompute', action='store_true',
                        help='Force recomputation, ignoring cache')
    parser.add_argument('--max-n', type=int, default=None,
                        help='Override max system size')
    parser.add_argument('--n-instances', type=int, default=None,
                        help='Override number of random instances')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for PDFs')
    args = parser.parse_args()

    setup_plot_style()
    out_dir = Path(args.output_dir) if args.output_dir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = ['1', '3', '4'] if args.figure == 'all' else [args.figure]

    if '1' in figs:
        print("\n=== Figure 1: TFI Model ===")
        max_n = args.max_n or 10
        sizes = list(range(4, max_n + 1))
        data = generate_tfi_data(sizes, recompute=args.recompute)
        plot_figure1(data, save_path=out_dir / 'plot1_reproduced.pdf')

    if '3' in figs:
        print("\n=== Figure 3: Random Local Hamiltonians ===")
        max_n = args.max_n or 8
        sizes = list(range(3, max_n + 1))
        n_inst = args.n_instances or 50
        data = generate_random_data(sizes, num_instances=n_inst,
                                    recompute=args.recompute)
        plot_figure3(data, save_path=out_dir / 'plot3_reproduced.pdf')

    if '4' in figs:
        print("\n=== Figure 4: Gate Set Dependence ===")
        n_inst = args.n_instances or 50
        data = generate_gateset_data(n=5, num_instances=n_inst,
                                     recompute=args.recompute)
        plot_figure4(data, save_path=out_dir / 'plot4_reproduced.pdf')

    print("\nDone.")


if __name__ == '__main__':
    main()
