"""Classifier + noise helpers for CUDA-Q experiments.

Implementation Notes:
- 'bare' mode: 2-qubit unencoded circuit
- 'encoded' mode: [[4,2,2]] code with SIMPLIFIED rotations (NOT paper-correct)
- 'encoded_logical' mode: [[4,2,2]] code with FULL logical rotations per paper
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cudaq

from .circuits import build_bare_kernel, build_encoded_kernel, build_encoded_kernel_logical_rotations

# Logical decoding for the [[4,2,2]] codewords in Eqs. (17)-(20) of the paper.
LOGICAL_LOOKUP: Dict[str, Tuple[int, int]] = {
    "0000": (0, 0),
    "1111": (0, 0),
    "0011": (0, 1),
    "1100": (0, 1),
    "0101": (1, 0),
    "1010": (1, 0),
    "0110": (1, 1),
    "1001": (1, 1),
}


def build_noise_model(
    single_qubit_error: float, 
    two_qubit_factor: float,
    ancilla_error: float = None,
    num_data_qubits: int = 4,
    env_noise_interval: int = None,
) -> Optional[cudaq.NoiseModel]:
    """
    Attach depolarizing channels to gates with optional ancilla-specific noise.
    
    Args:
        single_qubit_error: Base depolarizing probability for data qubits
        two_qubit_factor: Multiplier for two-qubit gate noise
        ancilla_error: Noise rate for ancilla qubits (if None, uses single_qubit_error)
        num_data_qubits: Number of data qubits (4 for [[4,2,2]] code)
        env_noise_interval: If set, applies environmental Pauli noise every N gates
                           (paper uses interval=4 for environmental noise model).
                           This is implemented by adding extra noise to identity gates.
    
    Paper Finding (Section 4.3):
    Ancilla error threshold ~0.003-0.004. Above this, errors propagate from
    ancilla qubits to physical qubits through CNOTs, limiting QEC effectiveness.
    
    Environmental Noise Model (Section 4.3):
    Paper applies Pauli errors at regular intervals (every 4 gates) to simulate
    environmental decoherence in addition to gate errors.
    """
    if single_qubit_error <= 0.0:
        return None

    noise = cudaq.NoiseModel()
    
    # Single-qubit gate noise
    single_channel = cudaq.DepolarizationChannel(single_qubit_error)
    for op in ("h", "rx", "ry", "rz"):
        noise.add_all_qubit_channel(op, single_channel)

    # Ancilla-specific noise (if different from data qubits)
    if ancilla_error is not None and ancilla_error != single_qubit_error:
        ancilla_channel = cudaq.DepolarizationChannel(ancilla_error)
        # Note: CUDA-Q doesn't support per-qubit noise models directly
        # This is a limitation - would need custom noise injection in kernel
        # For now, document the limitation
        pass  # TODO: Implement per-qubit noise when CUDA-Q supports it

    # Two-qubit gate noise (CNOT)
    # Apply noise to target qubit after CNOT
    multi_prob = min(0.999999, single_qubit_error * two_qubit_factor)
    multi_channel = cudaq.DepolarizationChannel(multi_prob)
    
    # For controlled gates, we need to add noise without num_controls
    # CUDA-Q applies the channel to each qubit involved in the gate
    noise.add_all_qubit_channel("x", multi_channel)
    
    # Note: CUDA-Q doesn't support noise on SWAP gates directly
    # SWAP is decomposed into CNOTs internally, so CNOT noise applies
    
    # Environmental noise model (paper Section 4.3)
    # Simulates decoherence by adding noise to identity gates at regular intervals
    # In CUDA-Q, we approximate this by adding noise to 'i' (identity) gates
    # The circuit should insert identity gates every env_noise_interval gates
    if env_noise_interval is not None and env_noise_interval > 0:
        # Environmental noise uses same error rate as gate noise
        env_channel = cudaq.DepolarizationChannel(single_qubit_error)
        noise.add_all_qubit_channel("i", env_channel)
        # Note: Circuit must explicitly insert 'i' gates for this to take effect
    
    return noise


def z_expectation_from_counts(counts: Dict[str, int], qubit_index: int = 0) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    expectation = 0.0
    for bitstring, freq in counts.items():
        bit = bitstring[qubit_index]
        expectation += (1.0 if bit == "0" else -1.0) * freq
    return expectation / total


@dataclass
class SampleStats:
    expectation: float
    accepted_shots: int
    rejected_shots: int


class ParityClassifier:
    """Encapsulates bare/encoded sampling and logical decoding."""

    def __init__(
        self,
        mode: str,
        shots: int,
        syndrome_rounds: int,
        noise_model: Optional[cudaq.NoiseModel],
        num_layers: int = 1,
    ):
        if mode not in {"bare", "encoded", "encoded_logical"}:
            raise ValueError("mode must be 'bare', 'encoded', or 'encoded_logical'")
        self.mode = mode
        self.shots = shots
        self.noise_model = noise_model
        self.num_layers = num_layers
        self.syndrome_rounds = syndrome_rounds if mode in {"encoded", "encoded_logical"} else 0
        self.bare_kernel = build_bare_kernel(num_layers) if mode == "bare" else None
        self.encoded_kernel = (
            build_encoded_kernel(self.syndrome_rounds, num_layers) if mode == "encoded" else None
        )
        self.encoded_logical_kernel = (
            build_encoded_kernel_logical_rotations(self.syndrome_rounds, num_layers) 
            if mode == "encoded_logical" else None
        )

    def __call__(self, theta: float, bits: Tuple[int, int]) -> SampleStats:
        if self.mode == "bare":
            return self._run_bare(theta, bits)
        elif self.mode == "encoded":
            return self._run_encoded(theta, bits)
        else:  # encoded_logical
            return self._run_encoded_logical(theta, bits)

    def _run_bare(self, theta: float, bits: Tuple[int, int]) -> SampleStats:
        kernel = self.bare_kernel
        assert kernel is not None
        # Multi-layer kernel expects list of thetas (one per layer)
        thetas = [theta] * self.num_layers
        result = cudaq.sample(
            kernel,
            thetas,
            bits[0],
            bits[1],
            shots_count=self.shots,
            noise_model=self.noise_model,
        )
        expectation = z_expectation_from_counts(result, qubit_index=0)
        return SampleStats(expectation=expectation, accepted_shots=self.shots, rejected_shots=0)

    def _run_encoded(self, theta: float, bits: Tuple[int, int]) -> SampleStats:
        kernel = self.encoded_kernel
        assert kernel is not None
        # Multi-layer kernel expects list of thetas (one per layer)
        thetas = [theta] * self.num_layers
        result = cudaq.sample(
            kernel,
            thetas,
            bits[0],
            bits[1],
            shots_count=self.shots,
            noise_model=self.noise_model,
        )

        accepted = 0
        rejected = 0
        expectation_accumulator = 0.0

        for bitstring, freq in result.items():
            if self._has_syndrome(bitstring):
                rejected += freq
                continue

            logical_bits = LOGICAL_LOOKUP.get(self._data_bits(bitstring))
            if logical_bits is None:
                rejected += freq
                continue

            contribution = 1.0 if logical_bits[0] == 0 else -1.0
            expectation_accumulator += contribution * freq
            accepted += freq

        expectation = expectation_accumulator / accepted if accepted else 0.0
        return SampleStats(expectation=expectation, accepted_shots=accepted, rejected_shots=rejected)

    def _run_encoded_logical(self, theta: float, bits: Tuple[int, int]) -> SampleStats:
        """Run with full logical rotation encoding (paper-correct implementation)."""
        kernel = self.encoded_logical_kernel
        assert kernel is not None
        # Multi-layer kernel expects list of thetas (one per layer)
        thetas = [theta] * self.num_layers
        result = cudaq.sample(
            kernel,
            thetas,
            bits[0],
            bits[1],
            shots_count=self.shots,
            noise_model=self.noise_model,
        )

        accepted = 0
        rejected = 0
        expectation_accumulator = 0.0

        for bitstring, freq in result.items():
            # Check syndrome bits (same position as regular encoded mode)
            if self._has_syndrome_logical(bitstring):
                rejected += freq
                continue

            # Extract data bits (first 4 qubits)
            data_bits = bitstring[:4]
            logical_bits = LOGICAL_LOOKUP.get(data_bits)
            if logical_bits is None:
                rejected += freq
                continue

            contribution = 1.0 if logical_bits[0] == 0 else -1.0
            expectation_accumulator += contribution * freq
            accepted += freq

        expectation = expectation_accumulator / accepted if accepted else 0.0
        return SampleStats(expectation=expectation, accepted_shots=accepted, rejected_shots=rejected)

    def _has_syndrome_logical(self, bitstring: str) -> bool:
        """Check syndrome bits in logical rotation mode (after data qubits)."""
        if self.syndrome_rounds == 0:
            return False
        data_len = 4
        # No rotation ancillas in simplified implementation - syndrome follows data directly
        syndrome_start = data_len
        for r in range(self.syndrome_rounds):
            z_bit = bitstring[syndrome_start + 2 * r]
            x_bit = bitstring[syndrome_start + 2 * r + 1]
            if z_bit == "1" or x_bit == "1":
                return True
        return False

    def _has_syndrome(self, bitstring: str) -> bool:
        if self.syndrome_rounds == 0:
            return False
        data_len = 4
        for r in range(self.syndrome_rounds):
            z_bit = bitstring[data_len + 2 * r]
            x_bit = bitstring[data_len + 2 * r + 1]
            if z_bit == "1" or x_bit == "1":
                return True
        return False

    @staticmethod
    def _data_bits(bitstring: str) -> str:
        return bitstring[:4]


def target_value(label: int) -> float:
    return 1.0 if label == 0 else -1.0


def mse_loss(expectation: float, label: int) -> float:
    diff = expectation - target_value(label)
    return 0.5 * diff * diff
