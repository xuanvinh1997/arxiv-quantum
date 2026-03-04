"""
Shadow Enhanced Greedy Quantum Eigensolver (SEGQE) — Qiskit Implementation

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

from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli, random_clifford
from qiskit.circuit import QuantumCircuit, Parameter


# ──────────────────────────────────────────────────────────
# Classical Shadow Machinery
# ──────────────────────────────────────────────────────────

def random_pauli_measurement(statevector: Statevector, num_qubits: int, rng: np.random.Generator):
    """
    Perform a single random Pauli measurement on a statevector.

    For each qubit, randomly choose a measurement basis (X, Y, or Z),
    apply the corresponding basis rotation, measure in computational basis,
    and return the measurement basis choice and outcome.

    Returns:
        bases: array of ints (0=X, 1=Y, 2=Z) for each qubit
        outcome: array of ints (0 or 1) for each qubit
    """
    # Choose random Pauli basis for each qubit: 0=X, 1=Y, 2=Z
    bases = rng.integers(0, 3, size=num_qubits)

    # Build the basis-rotation circuit
    qc = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        if bases[q] == 0:      # X basis: apply H
            qc.h(q)
        elif bases[q] == 1:    # Y basis: apply S†H
            qc.sdg(q)
            qc.h(q)
        # Z basis: no rotation needed

    # Apply the rotation and get the new statevector
    rotated_sv = statevector.evolve(qc)

    # Sample from the computational basis
    probs = np.abs(rotated_sv.data) ** 2
    idx = rng.choice(len(probs), p=probs)
    outcome = np.array([(idx >> q) & 1 for q in range(num_qubits)])

    return bases, outcome


def estimate_pauli_expectation_from_shadows(
    shadows: list[tuple[np.ndarray, np.ndarray]],
    pauli_op: Pauli,
    num_qubits: int,
) -> float:
    """
    Estimate ⟨P⟩ from classical shadows using the median-of-means approach.

    For a k-local Pauli P, the single-shot estimator is:
      ĥat{p} = 3^k * ⟨b|U P U†|b⟩
    which equals ±3^k if the measurement basis is compatible with P,
    and 0 otherwise.

    Args:
        shadows: list of (bases, outcome) tuples
        pauli_op: the Pauli operator to estimate
        num_qubits: number of qubits

    Returns:
        estimated expectation value
    """
    pauli_label = pauli_op.to_label()  # e.g., 'IXZY' (Qiskit: qubit 0 is rightmost)

    estimates = []
    for bases, outcome in shadows:
        val = 1.0
        compatible = True
        for q in range(num_qubits):
            p_char = pauli_label[num_qubits - 1 - q]  # Qiskit ordering
            if p_char == 'I':
                continue
            # Check compatibility: measurement basis must match Pauli
            # basis mapping: 0->X, 1->Y, 2->Z
            pauli_to_basis = {'X': 0, 'Y': 1, 'Z': 2}
            if bases[q] != pauli_to_basis[p_char]:
                compatible = False
                break
            # Eigenvalue: (-1)^outcome[q]
            val *= 3.0 * (1 - 2 * outcome[q])

        if compatible:
            estimates.append(val)
        else:
            estimates.append(0.0)

    return float(np.mean(estimates)) if estimates else 0.0


def collect_classical_shadows(
    statevector: Statevector,
    num_qubits: int,
    num_shadows: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collect N independent classical shadows from random Pauli measurements."""
    shadows = []
    for _ in range(num_shadows):
        bases, outcome = random_pauli_measurement(statevector, num_qubits, rng)
        shadows.append((bases, outcome))
    return shadows


# ──────────────────────────────────────────────────────────
# Gate Set Generation
# ──────────────────────────────────────────────────────────

def generate_pauli_rotation_generators(
    num_qubits: int,
    locality: int = 2,
    nearest_neighbor_only: bool = False,
) -> list[tuple[str, tuple[int, ...]]]:
    """
    Generate Pauli rotation gate generators of given locality.

    Each generator is a Pauli string X with X²=I that defines a rotation
    gate U(θ) = exp(-iθX/2).

    Args:
        num_qubits: number of qubits
        locality: maximum locality of generators (default 2)
        nearest_neighbor_only: if True, only include nearest-neighbor 2-qubit gates

    Returns:
        List of (pauli_label, qubit_indices) tuples.
        pauli_label is a string like 'XY' describing the non-identity part.
        qubit_indices is a tuple of qubit indices.
    """
    generators = []
    pauli_chars = ['X', 'Y', 'Z']

    for loc in range(1, locality + 1):
        # Generate all Pauli strings of this locality
        for pauli_combo in iter_product(pauli_chars, repeat=loc):
            pauli_label = ''.join(pauli_combo)

            # Generate all qubit subsets of this size
            if loc == 1:
                for q in range(num_qubits):
                    generators.append((pauli_label, (q,)))
            elif loc == 2:
                for q1 in range(num_qubits):
                    start_q2 = q1 + 1 if nearest_neighbor_only else q1 + 1
                    end_q2 = min(q1 + 2, num_qubits) if nearest_neighbor_only else num_qubits
                    for q2 in range(start_q2, end_q2):
                        generators.append((pauli_label, (q1, q2)))
            else:
                # General case for locality > 2
                from itertools import combinations
                for qubits in combinations(range(num_qubits), loc):
                    generators.append((pauli_label, qubits))

    return generators


def generator_to_sparse_pauli_op(
    pauli_label: str,
    qubit_indices: tuple[int, ...],
    num_qubits: int,
) -> SparsePauliOp:
    """Convert a generator (pauli_label, qubit_indices) to a SparsePauliOp."""
    full_label = ['I'] * num_qubits
    for char, idx in zip(pauli_label, qubit_indices):
        full_label[idx] = char
    # Qiskit uses reversed ordering: qubit 0 is rightmost
    return SparsePauliOp(Pauli(''.join(reversed(full_label))))


# ──────────────────────────────────────────────────────────
# Energy Difference Computation
# ──────────────────────────────────────────────────────────

def compute_A_B_from_shadows(
    shadows: list[tuple[np.ndarray, np.ndarray]],
    generator_op: SparsePauliOp,
    hamiltonian: SparsePauliOp,
    num_qubits: int,
) -> tuple[float, float]:
    """
    Estimate A and B for a Pauli rotation generator X from classical shadows.

    For Pauli generator X with X²=I:
      A = ⟨ψ|H|ψ⟩ - ⟨ψ|XHX|ψ⟩ = 2 Σ_{i: {X,P_i}=0} c_i ⟨ψ|P_i|ψ⟩
      B = ⟨ψ|i[X,H]|ψ⟩ = 2 Σ_{i: {X,P_i}=0} c_i ⟨ψ|iXP_i|ψ⟩

    Only Hamiltonian terms that anticommute with X contribute.
    """
    A_val = 0.0
    B_val = 0.0

    gen_pauli = generator_op.paulis[0]

    for coeff, pauli_term in zip(hamiltonian.coeffs, hamiltonian.paulis):
        c_i = coeff.real

        # Check if generator and Hamiltonian term anticommute
        # If they commute, P_i contributes nothing to A or B
        if gen_pauli.commutes(pauli_term):
            continue

        # A contribution: 2 * c_i * ⟨P_i⟩
        p_i_exp = estimate_pauli_expectation_from_shadows(shadows, pauli_term, num_qubits)
        A_val += 2.0 * c_i * p_i_exp

        # B contribution: 2 * c_i * ⟨iXP_i⟩
        # iXP_i is a Pauli (up to phase). Compute it:
        xp_product = gen_pauli.compose(pauli_term)
        # The phase from Pauli multiplication: gen_pauli * pauli_term = phase * result_pauli
        # We need i * X * P_i, so multiply phase by i
        phase = 1j * (-1j) ** xp_product.phase  # Pauli compose stores phase as power of -i
        result_pauli = Pauli((xp_product.z, xp_product.x))  # strip phase

        ixp_exp = estimate_pauli_expectation_from_shadows(shadows, result_pauli, num_qubits)
        B_val += 2.0 * c_i * (phase * ixp_exp).real

    return A_val, B_val


def compute_A_B_exact(
    statevector: Statevector,
    generator_op: SparsePauliOp,
    hamiltonian: SparsePauliOp,
) -> tuple[float, float]:
    """
    Compute exact A and B values (for validation/comparison).

    A = ⟨ψ|H|ψ⟩ - ⟨ψ|XHX|ψ⟩
    B = ⟨ψ|i[X,H]|ψ⟩
    """
    E = statevector.expectation_value(hamiltonian).real

    # XHX
    xhx = generator_op @ hamiltonian @ generator_op
    xhx = xhx.simplify()
    E_xhx = statevector.expectation_value(xhx).real
    A = E - E_xhx

    # i[X, H] = i(XH - HX)
    comm = generator_op @ hamiltonian - hamiltonian @ generator_op
    comm = (1j * comm).simplify()
    B = statevector.expectation_value(comm).real

    return A, B


# ──────────────────────────────────────────────────────────
# SEGQE Core
# ──────────────────────────────────────────────────────────

@dataclass
class SEGQEResult:
    """Result container for SEGQE."""
    circuit: QuantumCircuit = None
    energy: float = 0.0
    energy_history: list[float] = field(default_factory=list)
    gates_appended: list[tuple] = field(default_factory=list)
    num_iterations: int = 0
    final_statevector: Statevector = None


def segqe(
    hamiltonian: SparsePauliOp,
    num_qubits: int,
    initial_state: Optional[Statevector] = None,
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
        hamiltonian: The Hamiltonian as a SparsePauliOp
        num_qubits: Number of qubits
        initial_state: Initial statevector (default: |0...0⟩)
        max_depth: Maximum number of gates to append
        threshold: Minimum energy decrease to continue (Δ)
        num_shadows: Number of classical shadows per iteration (N)
        locality: Maximum locality of gate generators
        nearest_neighbor_only: Restrict to nearest-neighbor gates
        use_exact: Use exact statevector for A,B (True) or shadows (False)
        seed: Random seed
        verbose: Print progress

    Returns:
        SEGQEResult with circuit, energy history, etc.
    """
    rng = np.random.default_rng(seed)
    result = SEGQEResult()

    # Initial state
    if initial_state is None:
        initial_state = Statevector.from_label('0' * num_qubits)

    # Generate candidate gate generators
    generators = generate_pauli_rotation_generators(
        num_qubits, locality, nearest_neighbor_only
    )
    if verbose:
        print(f"SEGQE: {num_qubits} qubits, {len(generators)} candidate generators, "
              f"max_depth={max_depth}, threshold={threshold}")

    # Build circuit incrementally
    circuit = QuantumCircuit(num_qubits)
    current_state = initial_state.copy()

    # Compute initial energy
    E_current = current_state.expectation_value(hamiltonian).real
    result.energy_history.append(E_current)

    if verbose:
        print(f"  Iteration 0: E = {E_current:.8f}")

    for k in range(max_depth):
        # Collect classical shadows
        shadows = collect_classical_shadows(current_state, num_qubits, num_shadows, rng)

        best_dE = 0.0
        best_gen = None
        best_theta = None

        for pauli_label, qubit_indices in generators:
            gen_op = generator_to_sparse_pauli_op(pauli_label, qubit_indices, num_qubits)

            # Compute A, B
            if use_exact:
                A, B = compute_A_B_exact(current_state, gen_op, hamiltonian)
            else:
                A, B = compute_A_B_from_shadows(shadows, gen_op, hamiltonian, num_qubits)

            # Optimal parameters for Pauli rotation
            # ΔE(θ) = ½(A - A cos θ - B sin θ)
            # θ* = atan2(B, A)
            # ΔE_max = ½(A + √(A² + B²))
            theta_star = np.arctan2(B, A)
            dE_max = 0.5 * (A + np.sqrt(A**2 + B**2))

            if dE_max > best_dE:
                best_dE = dE_max
                best_gen = (pauli_label, qubit_indices)
                best_theta = theta_star

        # Check convergence
        if best_dE <= threshold:
            if verbose:
                print(f"  Converged: best ΔE = {best_dE:.2e} ≤ threshold {threshold}")
            break

        # Append the best gate: exp(-iθ*/2 · X)
        pauli_label, qubit_indices = best_gen
        gen_op = generator_to_sparse_pauli_op(pauli_label, qubit_indices, num_qubits)

        # Build rotation circuit for this gate
        gate_circuit = QuantumCircuit(num_qubits)
        # Pauli rotation: exp(-i θ/2 P)
        # Use Qiskit's built-in Pauli rotation
        pauli_str = gen_op.paulis[0].to_label()
        _append_pauli_rotation(gate_circuit, pauli_str, best_theta, num_qubits)

        circuit = circuit.compose(gate_circuit)
        current_state = initial_state.evolve(circuit)
        E_current = current_state.expectation_value(hamiltonian).real
        result.energy_history.append(E_current)
        result.gates_appended.append((best_gen, best_theta, best_dE))

        if verbose:
            print(f"  Iteration {k+1}: E = {E_current:.8f}, "
                  f"ΔE = {best_dE:.6f}, gate = {best_gen}")

    result.circuit = circuit
    result.energy = E_current
    result.num_iterations = len(result.gates_appended)
    result.final_statevector = current_state

    return result


def _append_pauli_rotation(
    circuit: QuantumCircuit,
    pauli_str: str,
    theta: float,
    num_qubits: int,
):
    """
    Append exp(-iθ/2 P) to the circuit, where P is a Pauli string.

    Decomposition: diagonalize P with single-qubit gates, then use
    a chain of CNOTs + Rz.
    """
    # Find active qubits and their Pauli types
    active_qubits = []
    pauli_types = []
    for q in range(num_qubits):
        p_char = pauli_str[num_qubits - 1 - q]  # Qiskit reversed ordering
        if p_char != 'I':
            active_qubits.append(q)
            pauli_types.append(p_char)

    if len(active_qubits) == 0:
        return

    # Step 1: Change basis from Pauli eigenbasis to Z basis
    for q, p_type in zip(active_qubits, pauli_types):
        if p_type == 'X':
            circuit.h(q)
        elif p_type == 'Y':
            circuit.rx(np.pi / 2, q)

    # Step 2: CNOT cascade to compute parity
    for i in range(len(active_qubits) - 1):
        circuit.cx(active_qubits[i], active_qubits[i + 1])

    # Step 3: Rz rotation on the last qubit
    circuit.rz(theta, active_qubits[-1])

    # Step 4: Undo CNOT cascade
    for i in range(len(active_qubits) - 2, -1, -1):
        circuit.cx(active_qubits[i], active_qubits[i + 1])

    # Step 5: Undo basis change
    for q, p_type in zip(active_qubits, pauli_types):
        if p_type == 'X':
            circuit.h(q)
        elif p_type == 'Y':
            circuit.rx(-np.pi / 2, q)


# ──────────────────────────────────────────────────────────
# Hamiltonian Constructors
# ──────────────────────────────────────────────────────────

def transverse_field_ising(
    num_qubits: int,
    w: float = 1.0,
    J: float = 1.0,
    periodic: bool = False,
) -> SparsePauliOp:
    """
    Construct the Transverse-Field Ising Hamiltonian.

    H = w Σ Z_i + J Σ X_i X_{i+1}

    Args:
        num_qubits: number of qubits/spins
        w: transverse field strength
        J: nearest-neighbor coupling
        periodic: use periodic boundary conditions
    """
    pauli_list = []
    coeffs = []

    # Single-qubit Z terms
    for i in range(num_qubits):
        label = ['I'] * num_qubits
        label[i] = 'Z'
        pauli_list.append(''.join(reversed(label)))
        coeffs.append(w)

    # Two-qubit XX terms
    n_bonds = num_qubits if periodic else num_qubits - 1
    for i in range(n_bonds):
        j = (i + 1) % num_qubits
        label = ['I'] * num_qubits
        label[i] = 'X'
        label[j] = 'X'
        pauli_list.append(''.join(reversed(label)))
        coeffs.append(J)

    return SparsePauliOp(pauli_list, coeffs=coeffs)


def random_local_hamiltonian(
    num_qubits: int,
    rng: Optional[np.random.Generator] = None,
) -> SparsePauliOp:
    """
    Construct a random local Hamiltonian (Eq. 5 in paper).

    H = Σ_i Σ_α w_i^α σ_i^α + Σ_i Σ_{α,β} J_i^{αβ} σ_i^α σ_{i+1}^β

    Periodic boundary conditions, coefficients drawn from N(0,1).
    """
    if rng is None:
        rng = np.random.default_rng()

    pauli_list = []
    coeffs = []
    paulis = ['X', 'Y', 'Z']

    # Single-qubit terms
    for i in range(num_qubits):
        for alpha in paulis:
            label = ['I'] * num_qubits
            label[i] = alpha
            pauli_list.append(''.join(reversed(label)))
            coeffs.append(rng.normal())

    # Two-qubit terms (periodic BC)
    for i in range(num_qubits):
        j = (i + 1) % num_qubits
        for alpha in paulis:
            for beta in paulis:
                label = ['I'] * num_qubits
                label[i] = alpha
                label[j] = beta
                pauli_list.append(''.join(reversed(label)))
                coeffs.append(rng.normal())

    return SparsePauliOp(pauli_list, coeffs=coeffs)


# ──────────────────────────────────────────────────────────
# Exact Diagonalization
# ──────────────────────────────────────────────────────────

def exact_ground_state_energy(hamiltonian: SparsePauliOp) -> tuple[float, np.ndarray]:
    """Compute exact ground state energy and vector via diagonalization."""
    H_matrix = hamiltonian.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    gs_energy = eigenvalues[0]
    gs_vector = eigenvectors[:, 0]
    return gs_energy, gs_vector


# ──────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────

def demo_tfi():
    """Demo: SEGQE on the Transverse-Field Ising model."""
    print("=" * 60)
    print("SEGQE Demo: Transverse-Field Ising Model (Qiskit)")
    print("=" * 60)

    for n_qubits in [4, 6]:
        for J_val, label in [(1.0, "critical (J=w=1)"), (0.5, "gapped (J=w/2)")]:
            print(f"\n--- n={n_qubits}, {label} ---")

            H = transverse_field_ising(n_qubits, w=1.0, J=J_val)
            E_exact, gs_vec = exact_ground_state_energy(H)
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
            fidelity = abs(np.vdot(gs_vec, result.final_statevector.data)) ** 2
            print(f"  SEGQE energy:     {result.energy:.8f}")
            print(f"  Relative error:   {rel_error:.6e}")
            print(f"  Fidelity:         {fidelity:.6f}")
            print(f"  Gates appended:   {result.num_iterations}")


def demo_random():
    """Demo: SEGQE on random local Hamiltonians."""
    print("\n" + "=" * 60)
    print("SEGQE Demo: Random Local Hamiltonians (Qiskit)")
    print("=" * 60)

    n_qubits = 4
    rng = np.random.default_rng(123)

    for trial in range(3):
        print(f"\n--- Trial {trial + 1}, n={n_qubits} ---")
        H = random_local_hamiltonian(n_qubits, rng)
        E_exact, gs_vec = exact_ground_state_energy(H)
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
        fidelity = abs(np.vdot(gs_vec, result.final_statevector.data)) ** 2
        print(f"  SEGQE energy:     {result.energy:.8f}")
        print(f"  Relative error:   {rel_error:.6e}")
        print(f"  Fidelity:         {fidelity:.6f}")
        print(f"  Gates appended:   {result.num_iterations}")


if __name__ == "__main__":
    demo_tfi()
    demo_random()
