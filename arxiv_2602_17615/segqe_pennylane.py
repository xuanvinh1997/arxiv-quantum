"""
Shadow Enhanced Greedy Quantum Eigensolver (SEGQE) — PennyLane Implementation

Reference: arXiv:2602.17615
"A Shadow Enhanced Greedy Quantum Eigensolver"

Algorithm overview:
  1. Start with C_0 = I, initial state |ψ_0⟩
  2. At each iteration k:
     a. Prepare |ψ_k⟩ = C_k|ψ_0⟩
     b. Collect N classical shadows via random Pauli measurements
     c. For each candidate gate U_j(θ) in gate set G:
        - Use shadows to estimate Pauli expectation values p_{j,α}
        - Classically compute ΔE_j(θ) and find optimal θ*_j
     d. Append the gate with largest energy decrease
  3. Terminate when ΔE_max ≤ Δ or circuit depth reaches D

For Pauli rotation gates e^{-iθX/2} with X²=I:
  ΔE(θ) = ½(A - A cos θ - B sin θ)
  θ* = atan2(B, A)
  ΔE_max = ½(A + √(A² + B²))
where A = ⟨ψ|H|ψ⟩ - ⟨ψ|XHX|ψ⟩, B = ⟨ψ|i[X,H]|ψ⟩
"""

import numpy as np
from itertools import product as iter_product
from typing import Optional
from dataclasses import dataclass, field

import pennylane as qml
from pennylane import numpy as pnp


# ──────────────────────────────────────────────────────────
# Classical Shadow Machinery
# ──────────────────────────────────────────────────────────

def random_pauli_measurement(
    state_vector: np.ndarray,
    num_qubits: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform a single random Pauli measurement on a state vector.

    For each qubit, randomly choose a measurement basis (X, Y, or Z),
    apply the corresponding basis rotation, measure in computational basis.

    Returns:
        bases: array of ints (0=X, 1=Y, 2=Z)
        outcome: array of ints (0 or 1) for each qubit
    """
    bases = rng.integers(0, 3, size=num_qubits)

    # Basis rotation matrices
    H_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    SdgH = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)  # S†H
    I_gate = np.eye(2)

    # Build full rotation as tensor product
    rotation = np.array([1.0])
    for q in range(num_qubits):
        if bases[q] == 0:    # X basis
            rotation = np.kron(rotation, H_gate)
        elif bases[q] == 1:  # Y basis
            rotation = np.kron(rotation, SdgH)
        else:                # Z basis
            rotation = np.kron(rotation, I_gate)

    rotated = rotation @ state_vector
    probs = np.abs(rotated) ** 2
    probs /= probs.sum()  # normalize for numerical safety

    idx = rng.choice(len(probs), p=probs)
    outcome = np.array([(idx >> q) & 1 for q in range(num_qubits)])

    return bases, outcome


def estimate_pauli_expectation(
    shadows: list[tuple[np.ndarray, np.ndarray]],
    pauli_word: dict[int, str],
    num_qubits: int,
) -> float:
    """
    Estimate ⟨P⟩ from classical shadows.

    Args:
        shadows: list of (bases, outcome) tuples
        pauli_word: dict mapping qubit_index -> 'X'/'Y'/'Z'
        num_qubits: number of qubits

    Returns:
        estimated expectation value
    """
    pauli_to_basis = {'X': 0, 'Y': 1, 'Z': 2}

    estimates = []
    for bases, outcome in shadows:
        val = 1.0
        compatible = True
        for q, p_type in pauli_word.items():
            if bases[q] != pauli_to_basis[p_type]:
                compatible = False
                break
            val *= 3.0 * (1 - 2 * outcome[q])

        estimates.append(val if compatible else 0.0)

    return float(np.mean(estimates)) if estimates else 0.0


def collect_classical_shadows(
    state_vector: np.ndarray,
    num_qubits: int,
    num_shadows: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collect N independent classical shadows from random Pauli measurements."""
    return [
        random_pauli_measurement(state_vector, num_qubits, rng)
        for _ in range(num_shadows)
    ]


# ──────────────────────────────────────────────────────────
# Pauli Algebra Utilities
# ──────────────────────────────────────────────────────────

# Pauli matrices
_PAULI = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}

# Pauli multiplication table: P_a * P_b = phase * P_c
_PAULI_MULT = {
    ('I', 'I'): (1, 'I'), ('I', 'X'): (1, 'X'), ('I', 'Y'): (1, 'Y'), ('I', 'Z'): (1, 'Z'),
    ('X', 'I'): (1, 'X'), ('X', 'X'): (1, 'I'), ('X', 'Y'): (1j, 'Z'), ('X', 'Z'): (-1j, 'Y'),
    ('Y', 'I'): (1, 'Y'), ('Y', 'X'): (-1j, 'Z'), ('Y', 'Y'): (1, 'I'), ('Y', 'Z'): (1j, 'X'),
    ('Z', 'I'): (1, 'Z'), ('Z', 'X'): (1j, 'Y'), ('Z', 'Y'): (-1j, 'X'), ('Z', 'Z'): (1, 'I'),
}


def pauli_commutes(
    pauli1: dict[int, str],
    pauli2: dict[int, str],
) -> bool:
    """Check if two Pauli operators commute."""
    anticommute_count = 0
    for q in set(pauli1.keys()) & set(pauli2.keys()):
        p1, p2 = pauli1[q], pauli2[q]
        if p1 != 'I' and p2 != 'I' and p1 != p2:
            anticommute_count += 1
    return anticommute_count % 2 == 0


def pauli_multiply(
    pauli1: dict[int, str],
    pauli2: dict[int, str],
) -> tuple[complex, dict[int, str]]:
    """
    Multiply two Pauli operators: P1 * P2 = phase * P_result.

    Returns (phase, result_pauli_word).
    """
    all_qubits = set(pauli1.keys()) | set(pauli2.keys())
    phase = 1.0 + 0j
    result = {}

    for q in all_qubits:
        p1 = pauli1.get(q, 'I')
        p2 = pauli2.get(q, 'I')
        ph, p_out = _PAULI_MULT[(p1, p2)]
        phase *= ph
        if p_out != 'I':
            result[q] = p_out

    return phase, result


# ──────────────────────────────────────────────────────────
# Gate Set Generation
# ──────────────────────────────────────────────────────────

def generate_pauli_rotation_generators(
    num_qubits: int,
    locality: int = 2,
    nearest_neighbor_only: bool = False,
) -> list[dict[int, str]]:
    """
    Generate Pauli rotation generators as dicts mapping qubit->Pauli.

    Each generator X with X²=I defines a rotation U(θ) = exp(-iθX/2).
    """
    generators = []
    pauli_chars = ['X', 'Y', 'Z']

    for loc in range(1, locality + 1):
        for pauli_combo in iter_product(pauli_chars, repeat=loc):
            if loc == 1:
                for q in range(num_qubits):
                    generators.append({q: pauli_combo[0]})
            elif loc == 2:
                for q1 in range(num_qubits):
                    end_q2 = min(q1 + 2, num_qubits) if nearest_neighbor_only else num_qubits
                    for q2 in range(q1 + 1, end_q2):
                        generators.append({q1: pauli_combo[0], q2: pauli_combo[1]})
            else:
                from itertools import combinations
                for qubits in combinations(range(num_qubits), loc):
                    gen = {q: p for q, p in zip(qubits, pauli_combo)}
                    generators.append(gen)

    return generators


# ──────────────────────────────────────────────────────────
# Energy Difference Computation
# ──────────────────────────────────────────────────────────

def hamiltonian_to_pauli_terms(
    hamiltonian: qml.Hamiltonian,
) -> list[tuple[float, dict[int, str]]]:
    """
    Convert a PennyLane Hamiltonian to a list of (coeff, pauli_word) tuples.
    """
    terms = []
    for coeff, obs in zip(hamiltonian.coeffs, hamiltonian.ops):
        pauli_word = _obs_to_pauli_word(obs)
        terms.append((float(np.real(coeff)), pauli_word))
    return terms


def _obs_to_pauli_word(obs) -> dict[int, str]:
    """Convert a PennyLane observable to a pauli_word dict."""
    if isinstance(obs, qml.Identity):
        return {}
    if isinstance(obs, (qml.PauliX, qml.PauliY, qml.PauliZ)):
        name_map = {'PauliX': 'X', 'PauliY': 'Y', 'PauliZ': 'Z'}
        return {obs.wires[0]: name_map[obs.name]}
    if isinstance(obs, qml.ops.Prod):
        result = {}
        for factor in obs.operands:
            pw = _obs_to_pauli_word(factor)
            result.update(pw)
        return result
    if isinstance(obs, qml.operation.Tensor):
        result = {}
        for factor in obs.obs:
            pw = _obs_to_pauli_word(factor)
            result.update(pw)
        return result
    raise ValueError(f"Unsupported observable type: {type(obs)}")


def compute_A_B_from_shadows(
    shadows: list[tuple[np.ndarray, np.ndarray]],
    generator: dict[int, str],
    ham_terms: list[tuple[float, dict[int, str]]],
    num_qubits: int,
) -> tuple[float, float]:
    """
    Estimate A and B from classical shadows for a Pauli generator X.

    A = 2 Σ_{i: {X,P_i}=0} c_i ⟨P_i⟩
    B = 2 Σ_{i: {X,P_i}=0} c_i ⟨iXP_i⟩
    """
    A_val = 0.0
    B_val = 0.0

    for c_i, pauli_term in ham_terms:
        if pauli_commutes(generator, pauli_term):
            continue

        # A contribution
        p_exp = estimate_pauli_expectation(shadows, pauli_term, num_qubits)
        A_val += 2.0 * c_i * p_exp

        # B contribution: need ⟨iXP_i⟩
        phase, xp_product = pauli_multiply(generator, pauli_term)
        # We want i * X * P_i, so multiply by i
        full_phase = 1j * phase
        ixp_exp = estimate_pauli_expectation(shadows, xp_product, num_qubits)
        B_val += 2.0 * c_i * (full_phase * ixp_exp).real

    return A_val, B_val


def compute_A_B_exact(
    state_vector: np.ndarray,
    generator: dict[int, str],
    ham_terms: list[tuple[float, dict[int, str]]],
    num_qubits: int,
) -> tuple[float, float]:
    """
    Compute exact A and B using the state vector.

    A = ⟨ψ|H|ψ⟩ - ⟨ψ|XHX|ψ⟩ = 2 Σ_{i: anticommute} c_i ⟨P_i⟩
    B = ⟨ψ|i[X,H]|ψ⟩ = 2 Σ_{i: anticommute} c_i ⟨iXP_i⟩
    """
    A_val = 0.0
    B_val = 0.0

    for c_i, pauli_term in ham_terms:
        if pauli_commutes(generator, pauli_term):
            continue

        # ⟨P_i⟩ exactly
        p_mat = _pauli_word_to_matrix(pauli_term, num_qubits)
        p_exp = (state_vector.conj() @ p_mat @ state_vector).real
        A_val += 2.0 * c_i * p_exp

        # ⟨iXP_i⟩ exactly
        phase, xp_product = pauli_multiply(generator, pauli_term)
        xp_mat = _pauli_word_to_matrix(xp_product, num_qubits)
        xp_exp = state_vector.conj() @ xp_mat @ state_vector
        B_val += 2.0 * c_i * (1j * phase * xp_exp).real

    return A_val, B_val


def _pauli_word_to_matrix(pauli_word: dict[int, str], num_qubits: int) -> np.ndarray:
    """Build the full 2^n × 2^n matrix for a Pauli word."""
    mat = np.array([1.0], dtype=complex)
    for q in range(num_qubits):
        p = pauli_word.get(q, 'I')
        mat = np.kron(mat, _PAULI[p])
    return mat


# ──────────────────────────────────────────────────────────
# Circuit Construction
# ──────────────────────────────────────────────────────────

def apply_pauli_rotation_to_state(
    state_vector: np.ndarray,
    generator: dict[int, str],
    theta: float,
    num_qubits: int,
) -> np.ndarray:
    """
    Apply exp(-iθ/2 P) to a state vector.

    Uses matrix exponentiation for exact simulation.
    """
    P_mat = _pauli_word_to_matrix(generator, num_qubits)
    # exp(-iθ/2 P) = cos(θ/2) I - i sin(θ/2) P
    rotation = np.cos(theta / 2) * np.eye(2**num_qubits) - 1j * np.sin(theta / 2) * P_mat
    return rotation @ state_vector


def build_pennylane_circuit(
    gates_appended: list[tuple[dict[int, str], float]],
    num_qubits: int,
):
    """
    Build a PennyLane QNode that applies the SEGQE circuit.

    Args:
        gates_appended: list of (generator_dict, theta) tuples
        num_qubits: number of qubits

    Returns:
        A function that constructs the PennyLane circuit operations.
    """
    def circuit():
        for generator, theta in gates_appended:
            _apply_pauli_rotation_pennylane(generator, theta, num_qubits)

    return circuit


def _apply_pauli_rotation_pennylane(
    generator: dict[int, str],
    theta: float,
    num_qubits: int,
):
    """Apply a Pauli rotation exp(-iθ/2 P) using PennyLane operations."""
    active_qubits = sorted(generator.keys())
    pauli_types = [generator[q] for q in active_qubits]

    if len(active_qubits) == 1:
        q = active_qubits[0]
        p = pauli_types[0]
        if p == 'X':
            qml.RX(theta, wires=q)
        elif p == 'Y':
            qml.RY(theta, wires=q)
        elif p == 'Z':
            qml.RZ(theta, wires=q)
        return

    # General multi-qubit Pauli rotation
    # Step 1: Basis rotation
    for q, p in zip(active_qubits, pauli_types):
        if p == 'X':
            qml.Hadamard(wires=q)
        elif p == 'Y':
            qml.RX(np.pi / 2, wires=q)

    # Step 2: CNOT cascade
    for i in range(len(active_qubits) - 1):
        qml.CNOT(wires=[active_qubits[i], active_qubits[i + 1]])

    # Step 3: Rz on last qubit
    qml.RZ(theta, wires=active_qubits[-1])

    # Step 4: Undo CNOT cascade
    for i in range(len(active_qubits) - 2, -1, -1):
        qml.CNOT(wires=[active_qubits[i], active_qubits[i + 1]])

    # Step 5: Undo basis rotation
    for q, p in zip(active_qubits, pauli_types):
        if p == 'X':
            qml.Hadamard(wires=q)
        elif p == 'Y':
            qml.RX(-np.pi / 2, wires=q)


# ──────────────────────────────────────────────────────────
# SEGQE Core
# ──────────────────────────────────────────────────────────

@dataclass
class SEGQEResult:
    """Result container for SEGQE."""
    energy: float = 0.0
    energy_history: list[float] = field(default_factory=list)
    gates_appended: list[tuple] = field(default_factory=list)
    num_iterations: int = 0
    final_state: np.ndarray = None
    circuit_ops: list = field(default_factory=list)


def segqe(
    hamiltonian: qml.Hamiltonian,
    num_qubits: int,
    initial_state: Optional[np.ndarray] = None,
    max_depth: int = 50,
    threshold: float = 1e-3,
    num_shadows: int = 5000,
    locality: int = 2,
    nearest_neighbor_only: bool = False,
    use_exact: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> SEGQEResult:
    """
    Shadow Enhanced Greedy Quantum Eigensolver (SEGQE).

    Args:
        hamiltonian: PennyLane Hamiltonian
        num_qubits: Number of qubits
        initial_state: Initial state vector (default: |0...0⟩)
        max_depth: Maximum number of gates to append
        threshold: Minimum energy decrease to continue (Δ)
        num_shadows: Number of classical shadows per iteration (N)
        locality: Maximum locality of gate generators
        nearest_neighbor_only: Restrict to nearest-neighbor gates
        use_exact: Use exact state vector for A,B (True) or shadows (False)
        seed: Random seed
        verbose: Print progress

    Returns:
        SEGQEResult
    """
    rng = np.random.default_rng(seed)
    result = SEGQEResult()

    # Initial state
    if initial_state is None:
        initial_state = np.zeros(2**num_qubits, dtype=complex)
        initial_state[0] = 1.0

    # Parse Hamiltonian into terms
    ham_terms = hamiltonian_to_pauli_terms(hamiltonian)

    # Generate candidate gate generators
    generators = generate_pauli_rotation_generators(
        num_qubits, locality, nearest_neighbor_only
    )
    if verbose:
        print(f"SEGQE: {num_qubits} qubits, {len(generators)} candidate generators, "
              f"max_depth={max_depth}, threshold={threshold}")

    current_state = initial_state.copy()

    # Compute initial energy
    H_mat = _hamiltonian_to_matrix(ham_terms, num_qubits)
    E_current = (current_state.conj() @ H_mat @ current_state).real
    result.energy_history.append(E_current)

    if verbose:
        print(f"  Iteration 0: E = {E_current:.8f}")

    for k in range(max_depth):
        # Collect classical shadows
        shadows = collect_classical_shadows(current_state, num_qubits, num_shadows, rng)

        best_dE = 0.0
        best_gen = None
        best_theta = None

        for gen in generators:
            # Compute A, B
            if use_exact:
                A, B = compute_A_B_exact(current_state, gen, ham_terms, num_qubits)
            else:
                A, B = compute_A_B_from_shadows(shadows, gen, ham_terms, num_qubits)

            # Optimal parameters for Pauli rotation
            # ΔE(θ) = ½(A - A cos θ - B sin θ)
            # Maximum at θ* = atan2(-B, -A)  [minimizes A cos θ + B sin θ]
            # ΔE_max = ½(A + √(A² + B²))
            theta_star = np.arctan2(-B, -A)
            dE_max = 0.5 * (A + np.sqrt(A**2 + B**2))

            if dE_max > best_dE:
                best_dE = dE_max
                best_gen = gen
                best_theta = theta_star

        # Check convergence
        if best_dE <= threshold:
            if verbose:
                print(f"  Converged: best ΔE = {best_dE:.2e} ≤ threshold {threshold}")
            break

        # Apply the best gate
        current_state = apply_pauli_rotation_to_state(
            current_state, best_gen, best_theta, num_qubits
        )
        E_current = (current_state.conj() @ H_mat @ current_state).real
        result.energy_history.append(E_current)
        result.gates_appended.append((best_gen, best_theta, best_dE))
        result.circuit_ops.append((best_gen, best_theta))

        gen_str = ''.join(f"{p}{q}" for q, p in sorted(best_gen.items()))
        if verbose:
            print(f"  Iteration {k+1}: E = {E_current:.8f}, "
                  f"ΔE = {best_dE:.6f}, gate = {gen_str}")

    result.energy = E_current
    result.num_iterations = len(result.gates_appended)
    result.final_state = current_state

    return result


def _hamiltonian_to_matrix(
    ham_terms: list[tuple[float, dict[int, str]]],
    num_qubits: int,
) -> np.ndarray:
    """Build the full Hamiltonian matrix."""
    dim = 2**num_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for c, pw in ham_terms:
        H += c * _pauli_word_to_matrix(pw, num_qubits)
    return H


# ──────────────────────────────────────────────────────────
# Hamiltonian Constructors
# ──────────────────────────────────────────────────────────

def transverse_field_ising(
    num_qubits: int,
    w: float = 1.0,
    J: float = 1.0,
    periodic: bool = False,
) -> qml.Hamiltonian:
    """
    Construct the Transverse-Field Ising Hamiltonian.

    H = w Σ Z_i + J Σ X_i X_{i+1}
    """
    coeffs = []
    obs = []

    # Single-qubit Z terms
    for i in range(num_qubits):
        coeffs.append(w)
        obs.append(qml.PauliZ(i))

    # Two-qubit XX terms
    n_bonds = num_qubits if periodic else num_qubits - 1
    for i in range(n_bonds):
        j = (i + 1) % num_qubits
        coeffs.append(J)
        obs.append(qml.PauliX(i) @ qml.PauliX(j))

    return qml.Hamiltonian(coeffs, obs)


def random_local_hamiltonian(
    num_qubits: int,
    rng: Optional[np.random.Generator] = None,
) -> qml.Hamiltonian:
    """
    Construct a random local Hamiltonian (Eq. 5 in paper).

    H = Σ_i Σ_α w_i^α σ_i^α + Σ_i Σ_{α,β} J_i^{αβ} σ_i^α σ_{i+1}^β
    Periodic BC, coefficients ~ N(0,1).
    """
    if rng is None:
        rng = np.random.default_rng()

    coeffs = []
    obs = []
    pauli_ops = {
        'X': qml.PauliX,
        'Y': qml.PauliY,
        'Z': qml.PauliZ,
    }

    # Single-qubit terms
    for i in range(num_qubits):
        for name, op_cls in pauli_ops.items():
            coeffs.append(float(rng.normal()))
            obs.append(op_cls(i))

    # Two-qubit terms (periodic BC)
    for i in range(num_qubits):
        j = (i + 1) % num_qubits
        for name1, op1 in pauli_ops.items():
            for name2, op2 in pauli_ops.items():
                coeffs.append(float(rng.normal()))
                obs.append(op1(i) @ op2(j))

    return qml.Hamiltonian(coeffs, obs)


# ──────────────────────────────────────────────────────────
# Exact Diagonalization
# ──────────────────────────────────────────────────────────

def exact_ground_state_energy(
    ham_terms: list[tuple[float, dict[int, str]]],
    num_qubits: int,
) -> tuple[float, np.ndarray]:
    """Compute exact ground state energy and vector."""
    H_mat = _hamiltonian_to_matrix(ham_terms, num_qubits)
    eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
    return float(eigenvalues[0]), eigenvectors[:, 0]


# ──────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────

def demo_tfi():
    """Demo: SEGQE on the Transverse-Field Ising model."""
    print("=" * 60)
    print("SEGQE Demo: Transverse-Field Ising Model (PennyLane)")
    print("=" * 60)

    for n_qubits in [4, 6]:
        for J_val, label in [(1.0, "critical (J=w=1)"), (0.5, "gapped (J=w/2)")]:
            print(f"\n--- n={n_qubits}, {label} ---")

            H = transverse_field_ising(n_qubits, w=1.0, J=J_val)
            ham_terms = hamiltonian_to_pauli_terms(H)
            E_exact, gs_vec = exact_ground_state_energy(ham_terms, n_qubits)
            print(f"  Exact ground state energy: {E_exact:.8f}")

            result = segqe(
                hamiltonian=H,
                num_qubits=n_qubits,
                max_depth=40,
                threshold=1e-3,
                num_shadows=5000,
                locality=2,
                nearest_neighbor_only=False,
                use_exact=True,
                seed=42,
                verbose=True,
            )

            rel_error = (E_exact - result.energy) / abs(E_exact)
            fidelity = abs(np.vdot(gs_vec, result.final_state)) ** 2
            print(f"  SEGQE energy:     {result.energy:.8f}")
            print(f"  Relative error:   {rel_error:.6e}")
            print(f"  Fidelity:         {fidelity:.6f}")
            print(f"  Gates appended:   {result.num_iterations}")

            # Show PennyLane circuit
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev)
            def segqe_circuit():
                circuit_fn = build_pennylane_circuit(result.circuit_ops, n_qubits)
                circuit_fn()
                return qml.expval(H)

            print(f"  PennyLane circuit energy: {segqe_circuit():.8f}")


def demo_random():
    """Demo: SEGQE on random local Hamiltonians."""
    print("\n" + "=" * 60)
    print("SEGQE Demo: Random Local Hamiltonians (PennyLane)")
    print("=" * 60)

    n_qubits = 4
    rng = np.random.default_rng(123)

    for trial in range(3):
        print(f"\n--- Trial {trial + 1}, n={n_qubits} ---")
        H = random_local_hamiltonian(n_qubits, rng)
        ham_terms = hamiltonian_to_pauli_terms(H)
        E_exact, gs_vec = exact_ground_state_energy(ham_terms, n_qubits)
        print(f"  Exact ground state energy: {E_exact:.8f}")

        result = segqe(
            hamiltonian=H,
            num_qubits=n_qubits,
            max_depth=40,
            threshold=1e-3,
            num_shadows=5000,
            locality=2,
            nearest_neighbor_only=False,
            use_exact=True,
            seed=42,
            verbose=True,
        )

        rel_error = (E_exact - result.energy) / abs(E_exact)
        fidelity = abs(np.vdot(gs_vec, result.final_state)) ** 2
        print(f"  SEGQE energy:     {result.energy:.8f}")
        print(f"  Relative error:   {rel_error:.6e}")
        print(f"  Fidelity:         {fidelity:.6f}")
        print(f"  Gates appended:   {result.num_iterations}")


if __name__ == "__main__":
    demo_tfi()
    demo_random()
