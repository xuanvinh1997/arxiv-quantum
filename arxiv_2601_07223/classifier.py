"""Classifier + noise helpers for CUDA-Q experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cudaq

from .circuits import build_bare_kernel, build_encoded_kernel

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


def build_noise_model(single_qubit_error: float, two_qubit_factor: float) -> Optional[cudaq.NoiseModel]:
    """Attach depolarizing channels to the gates referenced in the circuits."""
    if single_qubit_error <= 0.0:
        return None

    noise = cudaq.NoiseModel()
    single_channel = cudaq.DepolarizationChannel(single_qubit_error)
    for op in ("h", "rx", "ry", "rz"):
        noise.add_all_qubit_channel(op, single_channel)

    multi_prob = min(0.999999, single_qubit_error * two_qubit_factor)
    multi_channel = cudaq.DepolarizationChannel(multi_prob)
    noise.add_all_qubit_channel("x", multi_channel, num_controls=1)
    noise.add_all_qubit_channel("swap", multi_channel)
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
    ):
        if mode not in {"bare", "encoded"}:
            raise ValueError("mode must be 'bare' or 'encoded'")
        self.mode = mode
        self.shots = shots
        self.noise_model = noise_model
        self.syndrome_rounds = syndrome_rounds if mode == "encoded" else 0
        self.bare_kernel = build_bare_kernel() if mode == "bare" else None
        self.encoded_kernel = (
            build_encoded_kernel(self.syndrome_rounds) if mode == "encoded" else None
        )

    def __call__(self, theta: float, bits: Tuple[int, int]) -> SampleStats:
        if self.mode == "bare":
            return self._run_bare(theta, bits)
        return self._run_encoded(theta, bits)

    def _run_bare(self, theta: float, bits: Tuple[int, int]) -> SampleStats:
        kernel = self.bare_kernel
        assert kernel is not None
        result = cudaq.sample(
            kernel,
            theta,
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
        result = cudaq.sample(
            kernel,
            theta,
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
