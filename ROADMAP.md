# Implementation Roadmap: arXiv:2601.07223

## Current Status (January 29, 2026)

### ✅ Completed
- [x] [[4,2,2]] error detection code
- [x] Logical rotation encoding (paper-correct)
- [x] Multi-layer bare circuits
- [x] Syndrome extraction and shot rejection
- [x] Parameter-shift gradient descent
- [x] Noise model infrastructure
- [x] Comprehensive verification report

### 🚧 In Progress
- [ ] None currently

### 📋 Planned Features

---

## Phase 1: Deep Circuit Support (Priority: HIGH)
**Target**: Weeks 1-3 | **Status**: Not Started

| Task | Owner | Status | ETA | Blockers |
|------|-------|--------|-----|----------|
| Multi-layer encoded kernel | - | ⬜ Not Started | Week 1 | None |
| Multi-parameter classifier | - | ⬜ Not Started | Week 1 | None |
| Multi-parameter trainer | - | ⬜ Not Started | Week 2 | Classifier |
| CLI integration | - | ⬜ Not Started | Week 2 | Trainer |
| Testing (10-layer circuits) | - | ⬜ Not Started | Week 3 | CLI |
| Testing (75-layer circuits) | - | ⬜ Not Started | Week 3 | Testing |

**Deliverables**:
- Multi-layer circuits functional for all modes
- Training converges for 10-20 layer circuits
- Documentation updated

---

## Phase 2: Partial QEC Protocol (Priority: HIGH)
**Target**: Weeks 4-7 | **Status**: Not Started

| Task | Owner | Status | ETA | Blockers |
|------|-------|--------|-----|----------|
| Research Clifford+T algorithms | - | ⬜ Not Started | Week 4 | None |
| Gate decomposition module | - | ⬜ Not Started | Week 4-5 | Research |
| Partial QEC noise model | - | ⬜ Not Started | Week 5 | Decomposition |
| Partial QEC circuit builder | - | ⬜ Not Started | Week 6 | Noise model |
| CLI integration | - | ⬜ Not Started | Week 6 | Circuit |
| Validation vs. paper | - | ⬜ Not Started | Week 7 | CLI |

**Deliverables**:
- `partial_qec` mode functional
- Trainability at p=1.99×10⁻³ demonstrated
- Comparison with full QEC (simulated)

**Critical Dependencies**:
- External library for Clifford+T synthesis (PyZX or custom)
- CUDA-Q support for per-gate noise (or workaround)

---

## Phase 3: Ancilla Noise Support (Priority: MEDIUM)
**Target**: Weeks 8-10 | **Status**: Not Started

| Task | Owner | Status | ETA | Blockers |
|------|-------|--------|-----|----------|
| Investigate CUDA-Q noise APIs | - | ⬜ Not Started | Week 8 | None |
| Implement kernel-level noise | - | ⬜ Not Started | Week 9 | API research |
| OR implement workaround | - | ⬜ Not Started | Week 9 | API research |
| Ancilla threshold experiments | - | ⬜ Not Started | Week 10 | Implementation |
| Validation vs. paper | - | ⬜ Not Started | Week 10 | Experiments |

**Deliverables**:
- Ancilla-specific noise rates functional
- Threshold ~0.003-0.004 reproduced
- Documentation of limitations

**Risk**: High - CUDA-Q API may not support this feature

---

## Phase 4: MNIST Amplitude Encoding (Priority: LOW)
**Target**: Weeks 11-13 | **Status**: Not Started

| Task | Owner | Status | ETA | Blockers |
|------|-------|--------|-----|----------|
| Amplitude encoding circuit | - | ⬜ Not Started | Week 11 | None |
| 10-qubit VQC builder | - | ⬜ Not Started | Week 11-12 | Encoding |
| Dataset loader update | - | ⬜ Not Started | Week 12 | None |
| Classifier integration | - | ⬜ Not Started | Week 12 | VQC |
| Validation (small scale) | - | ⬜ Not Started | Week 13 | Classifier |

**Deliverables**:
- 10-qubit MNIST classifier functional
- Validation on 5-10 layer circuits (computational limit)
- Documentation of scalability limitations

**Computational Barrier**: Full 75-100 layer, 10-qubit circuits intractable

---

## Phase 5: Resource Estimation (Priority: LOW)
**Target**: Weeks 14-15 | **Status**: Not Started

| Task | Owner | Status | ETA | Blockers |
|------|-------|--------|-----|----------|
| Azure Quantum setup | - | ⬜ Not Started | Week 14 | Credentials |
| Resource estimator API | - | ⬜ Not Started | Week 14 | Setup |
| Circuit translation | - | ⬜ Not Started | Week 14-15 | API |
| Run estimations | - | ⬜ Not Started | Week 15 | Translation |
| Results analysis | - | ⬜ Not Started | Week 15 | Estimations |

**Deliverables**:
- Resource estimates for 10-qubit, 100-layer circuits
- Comparison: Full QEC vs. Partial QEC
- Validation of paper's 1.76×10⁶ qubits finding

**Prerequisites**: Azure Quantum workspace access

---

## Milestones

### M1: Deep Circuits Functional (Week 3)
- ✅ Success Criteria:
  - 10-layer circuits train successfully
  - Memory usage <32GB for 20 layers
  - Convergence behavior matches single-layer

### M2: Partial QEC Validated (Week 7)
- ✅ Success Criteria:
  - Training succeeds at p=1.99×10⁻³
  - Fails without partial QEC at same noise
  - T gate count matches paper's formula

### M3: Ancilla Threshold Reproduced (Week 10)
- ✅ Success Criteria:
  - Threshold behavior observed
  - Threshold value ~0.003-0.004
  - Zero ancilla noise gives ideal performance

### M4: Full Paper Coverage (Week 15)
- ✅ Success Criteria:
  - All sections implemented or documented as infeasible
  - Validation report shows agreement with paper
  - Clear documentation of any discrepancies

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CUDA-Q doesn't support per-gate noise | High | High | Implement statistical workaround |
| Clifford+T synthesis too complex | Medium | High | Use PyZX or simplified decomposition |
| Deep circuits exceed memory | High | High | Limit validation to 10-20 layers |
| Azure Quantum unavailable | Medium | Low | Use mock resource estimates |
| Training doesn't converge for deep circuits | Medium | Medium | Use Adam optimizer, better initialization |

---

## Resource Requirements

### Computational
- **Phase 1**: 16-32GB RAM, CPU-based acceptable
- **Phase 2**: 32-64GB RAM, GPU recommended
- **Phase 3**: 16-32GB RAM, CPU acceptable
- **Phase 4**: 64-128GB RAM, GPU strongly recommended
- **Phase 5**: Azure Quantum credits (cloud-based)

### Human Resources
- **Core development**: 1-2 developers
- **Quantum expertise**: 1 quantum algorithm specialist (Phases 2, 4)
- **Infrastructure**: 1 DevOps (Azure integration)

### Time
- **Minimum viable**: 7 weeks (Phases 1-2 only)
- **Full implementation**: 15 weeks (all phases)
- **With parallel work**: 10 weeks (2 developers)

---

## Version Control Strategy

### Branches
- `main`: Stable, verified implementations
- `dev/phase-1-deep-circuits`: Phase 1 work
- `dev/phase-2-partial-qec`: Phase 2 work
- `dev/phase-3-ancilla-noise`: Phase 3 work
- `dev/phase-4-amplitude-encoding`: Phase 4 work
- `dev/phase-5-resource-estimation`: Phase 5 work

### Merge Policy
- All merges require:
  - Unit tests passing
  - Documentation updated
  - Code review (if team >1)
  - Validation against paper

---

## Testing Strategy

### Unit Tests
- Circuit builders return valid kernels
- Noise models have correct parameters
- Gradient calculations match analytical values
- Logical decoding maps all 16 states correctly

### Integration Tests
- End-to-end training runs complete
- Multi-layer circuits produce sensible results
- Different modes (bare, encoded, partial_qec) are compatible

### Validation Tests
- Compare results with paper's figures/tables
- Reproduce threshold experiments
- Verify resource estimates (when available)

---

## Documentation Requirements

### Code Documentation
- Docstrings for all public functions
- Type hints throughout
- Examples in docstrings

### User Documentation
- Updated README with new features
- Tutorial notebooks for each mode
- Troubleshooting guide

### Research Documentation
- Validation report for each phase
- Discrepancy analysis
- Limitations and future work

---

## Communication Plan

### Weekly Updates
- Progress against milestones
- Blockers and mitigation
- Resource needs

### Phase Completion Reports
- What was delivered
- What changed from plan
- Lessons learned

### Final Deliverable
- Complete implementation status
- Validation results
- Future recommendations

---

## Success Metrics

### Quantitative
- [ ] 75-100 layer circuits functional
- [ ] Training at p=1.99×10⁻³ (partial QEC)
- [ ] Ancilla threshold at 0.003±0.001
- [ ] Resource estimates within 10% of paper
- [ ] Test coverage >80%

### Qualitative
- [ ] Code maintainable and well-documented
- [ ] Clear explanation of all limitations
- [ ] Reproducible results
- [ ] Community-ready (if open-sourcing)

---

## Next Actions

### This Week
1. Review and approve implementation plan
2. Set up development branches
3. Identify resource availability
4. Prioritize phases based on goals

### Next Week
1. Begin Phase 1 implementation
2. Set up CI/CD pipeline
3. Create unit test framework
4. Start weekly progress tracking

---

## Appendix: Quick Start

### For Phase 1 Only (Minimum Viable)
```bash
# Create branch
git checkout -b dev/phase-1-deep-circuits

# Implement multi-layer kernels
# ... development work ...

# Test
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded \
    --num-layers 10 \
    --shots 2048

# Merge when ready
git checkout main
git merge dev/phase-1-deep-circuits
```

### For Full Implementation
Follow the phase-by-phase roadmap above, with testing and validation at each milestone.
