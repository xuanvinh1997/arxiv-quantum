# Quick Reference: arXiv:2601.07223 Implementation

## Status at a Glance

| Section | Feature | Status | Priority |
|---------|---------|--------|----------|
| § 4 | [[4,2,2]] Error Detection | ✅ Complete | - |
| § 4.2 | Logical Rotation Encoding | ✅ Complete | - |
| § 4 | Syndrome Extraction | ✅ Complete | - |
| § 4 | Multi-round Syndrome | ✅ Complete | - |
| § 3 | Partial QEC Protocol | ❌ Missing | 🔴 HIGH |
| § 3 | Deep Circuits (75-100 layers) | 🟡 Partial | 🔴 HIGH |
| § 3.1 | MNIST Amplitude Encoding | ❌ Missing | 🟢 LOW |
| § 2 | Resource Analysis | ❌ Missing | 🟢 LOW |
| § 4.3 | Ancilla-Specific Noise | 🟡 Partial | 🟡 MED |

**Legend**: ✅ Complete | 🟡 Partial | ❌ Missing | 🔴 HIGH | 🟡 MED | 🟢 LOW

---

## Commands Cheat Sheet

### Current Features

```bash
# Paper-correct error detection (Section 4)
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded_logical \
    --syndrome-rounds 2 \
    --gate-noise 0.005 \
    --shots 4096

# Fast testing (simplified rotations)
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded \
    --syndrome-rounds 1 \
    --shots 2048

# Compare all three modes
for mode in bare encoded encoded_logical; do
    python -m arxiv_2601_07223.cli \
        --dataset parity \
        --mode $mode \
        --shots 2048 \
        --log-json logs/${mode}.json
done

# Ancilla threshold experiment (limited - see IMPLEMENTATION_STATUS.md)
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded_logical \
    --gate-noise 0.003 \
    --ancilla-noise 0.004 \
    --syndrome-rounds 2 \
    --shots 4096
```

### Planned Features (After Implementation)

```bash
# Deep circuits (Phase 1 - after implementation)
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded \
    --num-layers 75 \
    --shots 4096

# Partial QEC (Phase 2 - after implementation)
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode partial_qec \
    --t-gate-error 1e-4 \
    --clifford-error 1e-8 \
    --num-layers 75 \
    --shots 4096

# MNIST amplitude encoding (Phase 4 - after implementation)
python -m arxiv_2601_07223.cli \
    --dataset mnist_amplitude \
    --mode encoded \
    --num-layers 10 \
    --shots 8192
```

---

## File Structure

```
arxiv-quantum/
├── README.md                    # Main documentation
├── SUMMARY.md                   # Executive summary
├── IMPLEMENTATION_STATUS.md     # Verification report
├── IMPLEMENTATION_PLAN.md       # Technical plan (15 weeks)
├── ROADMAP.md                   # Week-by-week schedule
├── requirements.txt             # Python dependencies
├── arxiv_2601_07223/           # Main package
│   ├── __init__.py
│   ├── circuits.py             # Quantum circuit builders
│   ├── classifier.py           # Classification logic
│   ├── cli.py                  # Command-line interface
│   ├── data.py                 # Dataset loaders
│   └── training.py             # Optimization loop
├── docs/
│   ├── cudaq-notes.md          # CUDA-Q tips
│   └── arXiv-2601.07223v1/     # Paper LaTeX source
└── plot_fig3.py                # Plotting utilities
```

---

## Circuit Modes

### `--mode bare`
- 2 qubits (unencoded)
- No error detection
- Baseline performance
- **Use for**: Comparison benchmark

### `--mode encoded`
- 4 data qubits + syndrome ancillas
- [[4,2,2]] code with **simplified rotations**
- ⚡ Fast (rotations on physical qubits)
- ❌ NOT fault-tolerant
- **Use for**: Quick testing

### `--mode encoded_logical`
- 4 data + 12 rotation + syndrome ancillas
- [[4,2,2]] code with **paper-correct rotations**
- ✅ Fault-tolerant (rotations on ancillas)
- Slower (more qubits)
- **Use for**: Paper-accurate experiments

### `--mode partial_qec` (Planned)
- Error-corrected Clifford gates
- Raw T gates (no distillation)
- Paper's main innovation
- **Use for**: Validating Section 3

---

## Key Parameters

| Parameter | Default | Paper Value | Description |
|-----------|---------|-------------|-------------|
| `--shots` | 2048 | 4096-8192 | Samples per training step |
| `--steps` | 80 | 100 | Gradient descent iterations |
| `--lr` | 0.2 | - | Learning rate |
| `--gate-noise` | 0.0 | 0.001-0.01 | Depolarizing probability |
| `--syndrome-rounds` | 1 | 1-4 | Stabilizer measurements |
| `--ancilla-noise` | None | 0.003-0.004 | Ancilla error rate |
| `--num-layers` | 1 | 75-100 | Variational layers |

---

## Paper Results to Reproduce

### Section 4: Error Detection ✅
- [x] [[4,2,2]] code detects single-qubit errors
- [x] Syndrome extraction via ancilla measurements
- [x] Training improves with error detection
- [x] Multiple syndrome rounds tested

**Current Support**: Fully implemented

### Section 4.3: Ancilla Threshold 🟡
- [ ] Ancilla error propagates to physical qubits
- [ ] Threshold error rate ~0.003-0.004
- [ ] Zero ancilla noise → ideal performance

**Current Support**: Partial (CLI parameter added, CUDA-Q limitation)

### Section 3: Partial QEC ❌
- [ ] Training at p=1.99×10⁻³ with partial QEC
- [ ] Fails without QEC at same noise
- [ ] 2 orders magnitude qubit reduction

**Current Support**: Not implemented (Phase 2)

### Section 2: Resource Analysis ❌
- [ ] ~1.76×10⁶ qubits for full QEC
- [ ] ~3×10⁴ qubits without distillation
- [ ] Code distance 15-17

**Current Support**: Not implemented (Phase 5)

---

## Common Issues

### Out of Memory
```bash
# Reduce shots
--shots 1024

# Reduce syndrome rounds
--syndrome-rounds 0

# Use bare mode
--mode bare
```

### Training Not Converging
```bash
# Increase learning rate
--lr 0.5

# More training steps
--steps 200

# Reduce noise
--gate-noise 0.001
```

### Slow Execution
```bash
# Use simplified mode
--mode encoded

# Reduce syndrome rounds
--syndrome-rounds 1

# Use GPU target (if available)
--target nvidia
```

---

## Implementation Timeline

### Completed ✅
- Week -2: Initial verification
- Week -1: Logical rotation encoding
- Week 0: Planning documents

### Planned 📅
- **Weeks 1-3**: Deep circuit support
- **Weeks 4-7**: Partial QEC protocol
- **Weeks 8-10**: Ancilla noise improvements
- **Weeks 11-13**: MNIST amplitude encoding
- **Weeks 14-15**: Resource estimation

See ROADMAP.md for detailed schedule.

---

## Decision Framework

### Choose Implementation Scope

**Option A: Minimum (7 weeks)**
- Phases 1-2 only
- Validates paper's core innovation
- Best for academic validation

**Option B: Comprehensive (15 weeks)**
- All phases
- Maximum alignment
- Best for full reproducibility

**Option C: Pragmatic (10 weeks)**
- Phases 1-3
- Critical experiments covered
- **Recommended**

---

## Getting Help

### Documentation
- **What's implemented?** → IMPLEMENTATION_STATUS.md
- **How to implement missing features?** → IMPLEMENTATION_PLAN.md
- **When to implement?** → ROADMAP.md
- **Why these choices?** → SUMMARY.md

### Quick Answers
- **Can I reproduce the paper?** → Partially (Section 4 only)
- **What's the main missing piece?** → Partial QEC (Section 3)
- **How long to full implementation?** → 15 weeks (or 7 for core)
- **Is the code correct?** → Yes for Section 4, pending for 2-3

---

## Key Findings from Verification

### ✅ Correct Implementations
1. [[4,2,2]] logical encoding matches paper's Eqs. 17-20
2. Stabilizer measurements (X and Z) correct
3. Syndrome extraction protocol matches paper
4. Logical rotation encoding follows paper's 6-step protocol
5. Noise model structure aligns with paper's approach

### ⚠️ Simplifications
1. 2-bit MNIST features vs. amplitude encoding
2. Single/dual-layer vs. 75-100 layers
3. Simplified rotation mode for speed

### ❌ Missing Critical Features
1. Partial QEC protocol (paper's main contribution)
2. Clifford+T decomposition
3. Deep circuit support for encoded modes
4. Per-qubit noise modeling

---

## Next Steps

### For Users
1. Test current implementation
2. Review IMPLEMENTATION_STATUS.md
3. Decide which missing features you need
4. Refer to IMPLEMENTATION_PLAN.md

### For Contributors
1. Review ROADMAP.md
2. Pick a phase to work on
3. Follow implementation plan
4. Submit PR with tests

### For Researchers
1. Use current code for Section 4 validation
2. Note limitations clearly in papers
3. Contribute to missing features
4. Cite both paper and implementation

---

## References

**Paper**: arXiv:2601.07223 (January 2026)
**Authors**: Adermann, Kang, Sevior, Usman
**Title**: "Quantum Error Correction and Detection for Quantum Machine Learning"

**Code**: This implementation
**Status**: Section 4 complete, Sections 2-3 planned
**License**: [To be specified]
