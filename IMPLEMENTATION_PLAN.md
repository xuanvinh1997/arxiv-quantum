# Implementation Plan: Missing Features from arXiv:2601.07223

## Overview

This document outlines a phased approach to implement the remaining features from the paper to achieve full alignment with all sections.

---

## Phase 1: Deep Circuit Support (HIGH PRIORITY)

### Goal
Enable 75-100 layer variational circuits for both bare and encoded modes to replicate paper's Section 3 experiments.

### Tasks

#### 1.1 Multi-Layer Encoded Circuits
**File**: `arxiv_2601_07223/circuits.py`

**Implementation**:
```python
def build_encoded_kernel_multilayer(syndrome_rounds: int, num_layers: int):
    """
    Multi-layer [[4,2,2]] circuit with configurable depth.
    
    Args:
        syndrome_rounds: Number of syndrome extraction rounds
        num_layers: Number of variational layers (1-100)
    
    Note: Each layer adds 6 rotations (3 per logical qubit)
    """
    # Calculate rotation ancillas: 2 logical qubits × 3 rotations × num_layers
    rotation_ancillas = 6 * num_layers
    syndrome_ancillas = 2 * syndrome_rounds
    
    @cudaq.kernel
    def kernel(thetas: list[float], bit0: int, bit1: int):
        # Validate input
        assert len(thetas) == num_layers, f"Expected {num_layers} parameters"
        
        data = cudaq.qvector(4)
        
        # State preparation
        h(data[0])
        x.ctrl(data[0], data[1])
        x.ctrl(data[0], data[2])
        x.ctrl(data[0], data[3])
        
        # Basis encoding
        if bit1:
            x(data[2])
            x(data[3])
        if bit0:
            x(data[1])
            x(data[3])
        
        # Variational layers
        for layer_idx in range(num_layers):
            theta = thetas[layer_idx]
            
            # Apply rotations to all 4 physical qubits
            for qubit_idx in range(4):
                rx(theta, data[qubit_idx])
                rz(theta, data[qubit_idx])
                ry(theta, data[qubit_idx])
            
            # Entangling gate (except last layer)
            if layer_idx < num_layers - 1:
                swap(data[0], data[1])
        
        # Syndrome extraction
        if syndrome_rounds > 0:
            anc = cudaq.qvector(syndrome_ancillas)
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
        else:
            mz(data)
    
    return kernel
```

**Estimated Effort**: 1-2 days
**Complexity**: Medium
**Blockers**: None

#### 1.2 Update Classifier for Multi-Parameter Support
**File**: `arxiv_2601_07223/classifier.py`

**Changes Needed**:
- Modify `ParityClassifier.__init__()` to accept `num_layers` parameter
- Update `_run_*()` methods to pass list of thetas instead of single theta
- Store `num_layers` as instance variable

**Estimated Effort**: 4-6 hours
**Complexity**: Low

#### 1.3 Update Trainer for Multi-Parameter Optimization
**File**: `arxiv_2601_07223/training.py`

**Changes Needed**:
- Modify `VariationalTrainer.train()` to handle `theta_vector: list[float]`
- Update `_compute_gradient()` to compute per-parameter gradients
- Implement parameter-shift rule for each theta independently
- Consider Adam optimizer for better convergence with many parameters

**Estimated Effort**: 1-2 days
**Complexity**: Medium

#### 1.4 CLI Support for Multi-Layer Experiments
**File**: `arxiv_2601_07223/cli.py`

**Changes**:
```python
parser.add_argument(
    "--num-layers",
    type=int,
    default=1,
    help="Number of variational layers (1-100). Paper uses 75-100 for deep circuits.",
)
parser.add_argument(
    "--theta-init",
    type=str,
    default="uniform",
    choices=["uniform", "normal", "zeros"],
    help="Initialization strategy for multi-parameter circuits.",
)
```

**Estimated Effort**: 2-3 hours
**Complexity**: Low

---

## Phase 2: Partial QEC Protocol (HIGH PRIORITY - Paper's Core Innovation)

### Goal
Implement Section 3's partial QEC where Clifford gates are error-corrected but T gates remain noisy.

### Background
- Paper's key finding: Partial QEC reduces qubit overhead by 2 orders of magnitude
- Trainable at noise level p=1.99×10⁻³ where full circuits fail
- Requires decomposing rotations into Clifford+T gates

### Tasks

#### 2.1 Clifford+T Decomposition
**New File**: `arxiv_2601_07223/gate_decomposition.py`

**Implementation**:
```python
"""Clifford+T decomposition for rotation gates."""

import numpy as np
from typing import List, Tuple

def decompose_rotation_to_clifford_t(
    gate_type: str,  # 'RX', 'RY', 'RZ'
    theta: float,
    precision: float = 1e-4,
) -> List[Tuple[str, float]]:
    """
    Decompose rotation gate into Clifford+T sequence.
    
    Paper references:
    - Precision ε determines number of T gates: ~log₂(1/ε)
    - Error rate: ε ≈ 1-(1-ε_T)^(log₂(1/ε)) where ε_T is raw T gate error
    
    Returns:
        List of (gate_name, angle) tuples
        Clifford gates: H, S, CNOT (no angle)
        T gates: T, T† (with error rate)
    """
    # Solovay-Kitaev algorithm or grid-based approximation
    # This is a placeholder - full implementation requires complex algorithm
    
    n_t_gates = int(np.ceil(np.log2(1.0 / precision)))
    
    # Simplified example: RZ(θ) ≈ H·RX(θ)·H
    # Full implementation would use proper Clifford+T synthesis
    sequence = [
        ('H', None),
        ('T', theta / (2 * n_t_gates)),  # Repeated n_t_gates times
        ('H', None),
    ]
    
    return sequence

def count_t_gates(circuit: List[Tuple[str, float]]) -> int:
    """Count T gates in decomposed circuit."""
    return sum(1 for gate, _ in circuit if gate in ('T', 'T_dag'))
```

**References**:
- Solovay-Kitaev algorithm
- Grid synth for Clifford+T synthesis
- Paper's formula: T gate consumption for precision ε

**Estimated Effort**: 1-2 weeks
**Complexity**: High
**Blockers**: Requires deep understanding of quantum compilation

#### 2.2 Selective Error Correction
**New File**: `arxiv_2601_07223/partial_qec.py`

**Implementation**:
```python
"""Partial QEC: Error-corrected Clifford gates, raw T gates."""

import cudaq
from typing import Optional

def build_partial_qec_noise_model(
    clifford_error: float,  # Very low (error-corrected)
    t_gate_error: float,    # Raw error rate ~1e-4
    two_qubit_factor: float = 2.0,
) -> cudaq.NoiseModel:
    """
    Build noise model for partial QEC protocol.
    
    Paper specification (Section 3):
    - Clifford gates (H, S, CNOT): Error-corrected, ε ≈ 1e-8
    - T gates: Raw error rate ε_T ≈ 1e-4 (no magic state distillation)
    - Overall rotation error: 1-(1-ε_T)^(log₂(1/ε))
    """
    noise = cudaq.NoiseModel()
    
    # Clifford gates: Very low error (post-correction)
    clifford_channel = cudaq.DepolarizationChannel(clifford_error)
    for gate in ('h', 's', 's_dag'):
        noise.add_all_qubit_channel(gate, clifford_channel)
    
    # Two-qubit Clifford (CNOT): Error-corrected
    cnot_channel = cudaq.DepolarizationChannel(clifford_error * two_qubit_factor)
    noise.add_all_qubit_channel('x', cnot_channel, num_controls=1)
    
    # T gates: Raw error rate (NO distillation)
    t_channel = cudaq.DepolarizationChannel(t_gate_error)
    noise.add_all_qubit_channel('t', t_channel)
    noise.add_all_qubit_channel('t_dag', t_channel)
    
    return noise

def build_partial_qec_circuit(num_layers: int, precision: float = 1e-4):
    """
    Build circuit using Clifford+T decomposition with partial QEC.
    
    Each rotation gate decomposed into:
    - Clifford gates (error-corrected)
    - T gates (raw, trainable)
    """
    from .gate_decomposition import decompose_rotation_to_clifford_t
    
    @cudaq.kernel
    def kernel(thetas: list[float], bit0: int, bit1: int):
        data = cudaq.qvector(2)
        
        # Basis encoding (Clifford)
        if bit0:
            x(data[0])
        if bit1:
            x(data[1])
        
        for layer_idx in range(num_layers):
            theta = thetas[layer_idx]
            
            # Decompose RX(θ) into Clifford+T
            rx_sequence = decompose_rotation_to_clifford_t('RX', theta, precision)
            for gate, angle in rx_sequence:
                if gate == 'H':
                    h(data[0])
                elif gate == 'T':
                    t(data[0])
                elif gate == 'T_dag':
                    tdg(data[0])
                # ... similar for other gates
            
            # CNOT (error-corrected Clifford)
            x.ctrl(data[0], data[1])
        
        mz(data)
    
    return kernel
```

**Estimated Effort**: 2-3 weeks
**Complexity**: Very High
**Blockers**: 
- Requires Clifford+T decomposition (Task 2.1)
- CUDA-Q may not support per-gate-type noise modeling

#### 2.3 CLI Integration
**File**: `arxiv_2601_07223/cli.py`

**Changes**:
```python
parser.add_argument(
    "--mode",
    choices=["bare", "encoded", "encoded_logical", "partial_qec"],
    default="encoded",
    help="Circuit variant including partial QEC from Section 3.",
)
parser.add_argument(
    "--t-gate-error",
    type=float,
    default=1e-4,
    help="Raw T gate error rate for partial QEC mode (no magic state distillation).",
)
parser.add_argument(
    "--clifford-error",
    type=float,
    default=1e-8,
    help="Error-corrected Clifford gate error rate for partial QEC mode.",
)
```

**Estimated Effort**: 1 day
**Complexity**: Low
**Dependency**: Requires Task 2.2

---

## Phase 3: Full Ancilla Noise Support (MEDIUM PRIORITY)

### Goal
Implement per-qubit noise models to replicate paper's ancilla error threshold experiments.

### Challenges
CUDA-Q doesn't natively support per-qubit noise. Two approaches:

#### 3.1 Approach A: Kernel-Level Noise Injection (Preferred)

**Implementation**:
```python
def inject_pauli_noise(qubit, error_rate: float):
    """Manually apply Pauli errors with probability error_rate."""
    # Requires random number generation in CUDA-Q kernel
    # May need to use quantum channels or custom gates
    pass

@cudaq.kernel
def kernel_with_ancilla_noise(
    theta: float, 
    bit0: int, 
    bit1: int,
    data_error: float,
    ancilla_error: float,
):
    data = cudaq.qvector(4)
    anc = cudaq.qvector(2)
    
    # ... circuit gates ...
    
    # After each gate on data qubits
    for q in range(4):
        inject_pauli_noise(data[q], data_error)
    
    # After each gate on ancilla qubits
    for a in range(2):
        inject_pauli_noise(anc[a], ancilla_error)
```

**Blockers**: 
- CUDA-Q may not support probabilistic operations in kernels
- Requires classical RNG integration

**Estimated Effort**: 1-2 weeks
**Complexity**: High

#### 3.2 Approach B: Multiple Noise Models (Workaround)

**Implementation**:
```python
def run_with_heterogeneous_noise(
    kernel,
    data_noise: float,
    ancilla_noise: float,
    shots: int,
):
    """
    Run circuit multiple times with different noise models and combine results.
    
    This is a statistical approximation:
    1. Run with data_noise on all qubits -> weight w1
    2. Run with ancilla_noise on all qubits -> weight w2
    3. Interpolate results based on qubit ratios
    """
    # Statistical approximation
    data_results = cudaq.sample(kernel, noise_model=data_noise_model, shots=shots)
    anc_results = cudaq.sample(kernel, noise_model=ancilla_noise_model, shots=shots)
    
    # Combine with weights proportional to qubit counts
    # This is NOT physically accurate but may give rough approximation
    pass
```

**Estimated Effort**: 3-5 days
**Complexity**: Medium
**Accuracy**: Low (statistical approximation only)

---

## Phase 4: MNIST Amplitude Encoding (LOW PRIORITY)

### Goal
Implement full 10-qubit amplitude encoding for 28×28 MNIST images as described in paper's Section 3.

### Tasks

#### 4.1 Amplitude Encoding Circuit
**New File**: `arxiv_2601_07223/amplitude_encoding.py`

**Implementation**:
```python
"""Amplitude encoding for classical data into quantum states."""

import numpy as np
import cudaq

def prepare_amplitude_encoded_state(image: np.ndarray, num_qubits: int = 10):
    """
    Encode 28×28 MNIST image into 2^10 = 1024 amplitude state.
    
    Paper approach:
    - Normalize pixel values to unit vector
    - Use state preparation circuit to load amplitudes
    - Requires O(2^n) gates for n qubits
    
    Args:
        image: 28×28 array, flattened to 784 pixels
        num_qubits: Number of qubits (10 for 1024 dimensions)
    """
    # Flatten and normalize
    pixels = image.flatten()[:2**num_qubits]  # Take first 1024 pixels
    norm = np.linalg.norm(pixels)
    amplitudes = pixels / norm if norm > 0 else pixels
    
    @cudaq.kernel
    def state_prep():
        qubits = cudaq.qvector(num_qubits)
        # CUDA-Q state preparation
        # This requires controlled rotation sequences
        # Full implementation is complex (multiplexed rotations)
        pass
    
    return state_prep

def build_10qubit_mnist_classifier(num_layers: int):
    """
    10-qubit VQC for MNIST classification with amplitude encoding.
    
    Paper specification (Section 3.1):
    - 10 qubits for 1024-dimensional state
    - 75-100 variational layers
    - Amplitude encoding of input
    - Full connectivity or hardware-aware layout
    """
    @cudaq.kernel
    def kernel(amplitudes: list[float], thetas: list[float]):
        qubits = cudaq.qvector(10)
        
        # State preparation (amplitude encoding)
        # ... complex multiplexed rotation circuit ...
        
        # Variational layers
        for layer in range(num_layers):
            # Apply rotations to all qubits
            for q in range(10):
                theta_idx = layer * 10 + q
                rx(thetas[theta_idx], qubits[q])
                rz(thetas[theta_idx], qubits[q])
                ry(thetas[theta_idx], qubits[q])
            
            # Entangling layer (all-to-all or nearest-neighbor)
            for q in range(9):
                x.ctrl(qubits[q], qubits[q + 1])
        
        mz(qubits)
    
    return kernel
```

**Estimated Effort**: 2-3 weeks
**Complexity**: Very High
**Computational Barrier**: 10-qubit density matrix simulation is intractable for 75-100 layers

#### 4.2 Dataset Loader Update
**File**: `arxiv_2601_07223/data.py`

**Changes**:
```python
def mnist_amplitude_encoded_dataset(
    digit_positive: int = 0,
    digit_negative: int = 1,
    limit_per_class: int = 128,
    data_dir: str = "./data",
    train_split: bool = True,
) -> list:
    """
    Load MNIST with amplitude encoding preparation.
    
    Returns:
        List of (amplitudes, label) where amplitudes is 1024-dim vector
    """
    # Load MNIST images
    # Normalize to unit vectors
    # Return amplitude arrays instead of 2-bit features
    pass
```

**Estimated Effort**: 1 week
**Complexity**: Medium

---

## Phase 5: Resource Analysis Integration (LOW PRIORITY)

### Goal
Integrate with Azure Quantum Resource Estimator to replicate Section 2's findings.

### Tasks

#### 5.1 Azure Integration
**New File**: `arxiv_2601_07223/resource_estimation.py`

**Implementation**:
```python
"""Azure Quantum Resource Estimator integration for Section 2."""

from azure.quantum import Workspace
from azure.quantum.qiskit import AzureQuantumProvider

def estimate_qec_resources(
    num_logical_qubits: int,
    num_layers: int,
    error_budget: float = 1e-4,
    code_distance: int = 15,
):
    """
    Estimate physical qubit requirements for full QEC.
    
    Paper findings (Section 2):
    - 10 logical qubits, 100 layers
    - Error budget 1e-4
    - Surface code with code distance 15-17
    - Magic state distillation for T gates
    - Result: ~1.76×10⁶ physical qubits
    
    Without distillation: ~120d² ≈ 27,000-34,000 qubits
    """
    # Connect to Azure Quantum workspace
    workspace = Workspace(
        resource_id="...",
        location="...",
    )
    
    # Define circuit for resource estimation
    # Run resource estimator
    # Parse results
    
    return {
        'physical_qubits': None,
        'logical_qubits': num_logical_qubits,
        'code_distance': code_distance,
        'magic_states': None,
        'runtime': None,
    }
```

**Estimated Effort**: 1-2 weeks
**Complexity**: Medium
**Prerequisites**: Azure Quantum account, API access

---

## Implementation Priorities

### Critical Path (Enables Paper Validation)
1. **Phase 1**: Deep circuit support → 2-3 weeks
2. **Phase 2**: Partial QEC protocol → 3-4 weeks
3. **Phase 3.1**: Kernel-level ancilla noise → 1-2 weeks

**Total: 6-9 weeks**

### Secondary Features
4. Phase 4: MNIST amplitude encoding → 2-3 weeks
5. Phase 5: Azure resource estimation → 1-2 weeks

**Total: 9-14 weeks overall**

---

## Risk Assessment

### High Risk Items
1. **Clifford+T decomposition**: Complex algorithm, may need external library
2. **Per-qubit noise in CUDA-Q**: API limitations may prevent full implementation
3. **Computational scalability**: 10-qubit, 100-layer circuits exceed classical simulation
4. **Azure integration**: Requires cloud access, potential API changes

### Mitigation Strategies
1. Use existing libraries (e.g., PyZX, Qiskit) for gate decomposition
2. Document CUDA-Q limitations clearly; propose workarounds
3. Focus on smaller-scale validation (e.g., 4-qubit, 10-layer circuits)
4. Make Azure integration optional with mock results

---

## Success Metrics

### Phase 1 Success
- [ ] Run 75-100 layer circuits successfully
- [ ] Training converges for deep circuits
- [ ] Memory usage remains tractable (<64GB RAM)

### Phase 2 Success
- [ ] Partial QEC mode reduces qubit overhead vs full QEC
- [ ] Training succeeds at p=1.99×10⁻³ noise level
- [ ] Untrainable without partial QEC (validates paper's claim)

### Phase 3 Success
- [ ] Ancilla error sweep shows threshold behavior
- [ ] Threshold ~0.003-0.004 matches paper
- [ ] Zero ancilla noise recovers ideal performance

### Phase 4 Success
- [ ] 10-qubit MNIST classification functional
- [ ] Accuracy comparable to paper's reported values
- [ ] Amplitude encoding verified correct

### Phase 5 Success
- [ ] Resource estimates match paper's 1.76×10⁶ qubits
- [ ] Partial QEC overhead 2 orders of magnitude lower
- [ ] Reproduce paper's Table/Figure on resource comparison

---

## Next Steps

### Immediate Actions
1. Set up development branches for each phase
2. Create unit tests for new modules
3. Benchmark current code on larger problems
4. Profile memory usage for deep circuits

### Week 1-2: Phase 1 Kickoff
- [ ] Implement multi-layer encoded kernel
- [ ] Update classifier for vector parameters
- [ ] Test with 10-layer circuits

### Week 3-6: Phase 2 Deep Dive
- [ ] Research Clifford+T synthesis algorithms
- [ ] Implement gate decomposition module
- [ ] Build partial QEC noise model
- [ ] Validate against paper's Section 3 results

### Month 2: Integration & Testing
- [ ] End-to-end testing of all modes
- [ ] Performance optimization
- [ ] Documentation updates
- [ ] Prepare validation report

---

## Questions for Clarification

1. **Computational resources**: What hardware is available? GPU access?
2. **Time constraints**: Is there a deadline for full implementation?
3. **Scope priorities**: Which missing features are most critical for your use case?
4. **External dependencies**: Can we use third-party libraries (PyZX, Cirq, Qiskit)?
5. **Azure access**: Do you have Azure Quantum workspace credentials?

---

## References

### Papers
- arXiv:2601.07223: Main paper
- Solovay-Kitaev algorithm for gate decomposition
- Surface code resource estimation papers

### Libraries
- CUDA-Q documentation
- PyZX for Clifford+T synthesis
- Azure Quantum SDK
- Qiskit for amplitude encoding

### Tools
- Quantum resource estimator
- Circuit optimization tools
- Noise simulation frameworks
