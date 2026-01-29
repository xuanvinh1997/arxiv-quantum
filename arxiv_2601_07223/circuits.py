"""CUDA-Q circuit builders for the [[4,2,2]] classifier."""

from __future__ import annotations

import cudaq


def build_bare_kernel():
    """Single-parameter, two-qubit classifier kernel."""

    @cudaq.kernel
    def kernel(theta: float, bit0: int, bit1: int):
        data = cudaq.qvector(2)
        if bit0:
            x(data[0])
        if bit1:
            x(data[1])

        rx(theta, data[0])
        rz(theta, data[0])
        ry(theta, data[0])
        x.ctrl(data[0], data[1])
        rx(theta, data[1])
        rz(theta, data[1])
        ry(theta, data[1])
        mz(data)

    return kernel


def build_encoded_kernel(syndrome_rounds: int):
    """Error-detecting classifier kernel with optional stabilizer rounds."""

    ancilla_per_round = 2 * syndrome_rounds

    if ancilla_per_round == 0:

        @cudaq.kernel
        def kernel(theta: float, bit0: int, bit1: int):
            data = cudaq.qvector(4)

            h(data[0])
            x.ctrl(data[0], data[1])
            x.ctrl(data[0], data[2])
            x.ctrl(data[0], data[3])
            if bit1:
                x(data[2])
                x(data[3])
            if bit0:
                x(data[1])
                x(data[3])

            for idx in range(4):
                rx(theta, data[idx])
                rz(theta, data[idx])
                ry(theta, data[idx])

            swap(data[0], data[1])

            for idx in range(4):
                rx(theta, data[idx])
                rz(theta, data[idx])
                ry(theta, data[idx])

            mz(data)

        return kernel

    @cudaq.kernel
    def kernel(theta: float, bit0: int, bit1: int):
        data = cudaq.qvector(4)
        anc = cudaq.qvector(ancilla_per_round)

        h(data[0])
        x.ctrl(data[0], data[1])
        x.ctrl(data[0], data[2])
        x.ctrl(data[0], data[3])
        if bit1:
            x(data[2])
            x(data[3])
        if bit0:
            x(data[1])
            x(data[3])

        for idx in range(4):
            rx(theta, data[idx])
            rz(theta, data[idx])
            ry(theta, data[idx])

        swap(data[0], data[1])

        for idx in range(4):
            rx(theta, data[idx])
            rz(theta, data[idx])
            ry(theta, data[idx])

        for round_idx in range(syndrome_rounds):
            anc_z = anc[2 * round_idx]
            anc_x = anc[2 * round_idx + 1]
            for q in range(4):
                x.ctrl(data[q], anc_z)
            for q in range(4):
                h(data[q])
            for q in range(4):
                x.ctrl(data[q], anc_x)
            for q in range(4):
                h(data[q])

        mz(data)
        mz(anc)

    return kernel
