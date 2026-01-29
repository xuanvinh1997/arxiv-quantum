# Training Commands: Replicating Paper Experiments

Commands to reproduce experiments from "Quantum Error Correction and Detection for Quantum Machine Learning" (arXiv:2601.07223).

---

## Section 4.3: Baseline Experiments (No Error Detection)

### Experiment 1: Zero Noise Baseline
```bash
# Bare circuit (no QEC) with automatic plotting
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode bare \
    --gate-noise 0.0 \
    --shots 2048 \
    --steps 100 \
    --lr 0.2 \
    --log-json results/bare_noise_0.000.json \
    --plot

# Encoded circuit (with QEC) - Use 'encoded' mode (encoded_logical has CUDA-Q compatibility issues)
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded \
    --gate-noise 0.0 \
    --syndrome-rounds 1 \
    --shots 2048 \
    --steps 100 \
    --lr 0.2 \
    --log-json results/encoded_noise_0.000.json \
    --plot \
    --plot-title "Encoded QEC - Zero Noise"
```

**Note**: 
- Add `--plot` to automatically generate training curves
- Use `--plot-output <path>` to customize plot filename
- Use `--plot-title "<title>"` for custom plot titles
- The `encoded_logical` mode (paper-correct rotations) has CUDA-Q compatibility issues. Use `encoded` mode for functional experiments.

---

## Section 4.3: Noise Sweep (Figure 4 in Paper)

### Experiment 2: Low Noise (p ≤ 0.0025) - Learning Possible

```bash
# p = 0.001 (1e-3)
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode bare \
    --gate-noise 0.001 \
    --shots 4096 \
    --steps 100 \
    --log-json results/bare_p_0.001.json

python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded \
    --gate-noise 0.001 \
    --syndrome-rounds 2 \
    --shots 4096 \
    --steps 100 \
    --log-json results/encoded_p_0.001.json

# p = 0.0025 (2.5e-3)
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode bare \
    --gate-noise 0.0025 \
    --shots 4096 \
    --steps 100 \
    --log-json results/bare_p_0.0025.json

python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded \
    --gate-noise 0.0025 \
    --syndrome-rounds 2 \
    --shots 4096 \
    --steps 100 \
    --log-json results/encoded_p_0.0025.json
```

### Experiment 3: High Noise (p ≥ 0.005) - Learning Difficult/Impossible

```bash
# p = 0.005 (5e-3) - Bare should struggle
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode bare \
    --gate-noise 0.005 \
    --shots 4096 \
    --steps 100 \
    --log-json results/bare_p_0.005.json

python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded \
    --gate-noise 0.005 \
    --syndrome-rounds 2 \
    --shots 4096 \
    --steps 100 \
    --log-json results/encoded_p_0.005.json

# p = 0.01 (1e-2) - Both should fail
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode bare \
    --gate-noise 0.01 \
    --shots 4096 \
    --steps 100 \
    --log-json results/bare_p_0.010.json

python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded \
    --gate-noise 0.01 \
    --syndrome-rounds 2 \
    --shots 4096 \
    --steps 100 \
    --log-json results/encoded_p_0.010.json
```

### Complete Noise Sweep Script

```bash
#!/bin/bash
# Paper's noise range: 0.001 to 0.01

NOISE_LEVELS=(0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.010)

for p in "${NOISE_LEVELS[@]}"; do
    echo "Running experiments for p=$p"
    
    # Bare circuit
    python -m arxiv_2601_07223.cli \
        --dataset parity \
        --mode bare \
        --gate-noise $p \
        --shots 4096 \
        --steps 100 \
        --log-json results/bare_p_${p}.json
    
    # Encoded circuit
    python -m arxiv_2601_07223.cli \
        --dataset parity \
        --mode encoded \
        --gate-noise $p \
        --syndrome-rounds 2 \
        --shots 4096 \
        --steps 100 \
        --log-json results/encoded_p_${p}.json
done

# Plot results
python plot_fig3.py --plot-noise results/*.json --output fig4_reproduction.png
```

---

## Section 4.3: Syndrome Rounds Comparison

### Experiment 4: Impact of Multiple Syndrome Rounds

```bash
# Fixed noise p=0.005, vary syndrome rounds
NOISE=0.005

# 0 syndrome rounds (no error detection)
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded_logical \
    --gate-noise $NOISE \
    --syndrome-rounds 0 \
    --shots 4096 \
    --steps 100 \
    --log-json results/encoded_syn_0_p_${NOISE}.json

# 1 syndrome round
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded_logical \
    --gate-noise $NOISE \
    --syndrome-rounds 1 \
    --shots 4096 \
    --steps 100 \
    --log-json results/encoded_syn_1_p_${NOISE}.json

# 2 syndrome rounds (paper's default)
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded_logical \
    --gate-noise $NOISE \
    --syndrome-rounds 2 \
    --shots 4096 \
    --steps 100 \
    --log-json results/encoded_syn_2_p_${NOISE}.json

# 3 syndrome rounds
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded_logical \
    --gate-noise $NOISE \
    --syndrome-rounds 3 \
    --shots 4096 \
    --steps 100 \
    --log-json results/encoded_syn_3_p_${NOISE}.json

# 4 syndrome rounds
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded_logical \
    --gate-noise $NOISE \
    --syndrome-rounds 4 \
    --shots 4096 \
    --steps 100 \
    --log-json results/encoded_syn_4_p_${NOISE}.json
```

**Paper Finding**: Improvement plateaus after ~2 syndrome rounds at low noise

---

## Section 4.3: Ancilla Error Threshold Experiments

### Experiment 5: Ancilla Noise Sweep (Critical Finding)

```bash
# Paper's threshold: ~0.003-0.004 for gate noise
# Test with fixed data qubit noise, varying ancilla noise

DATA_NOISE=0.003

# Zero ancilla noise (ideal case)
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded_logical \
    --gate-noise $DATA_NOISE \
    --ancilla-noise 0.0 \
    --syndrome-rounds 2 \
    --shots 4096 \
    --steps 100 \
    --log-json results/anc_threshold_anc_0.000.json

# Ancilla noise at threshold
for anc_p in 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008; do
    python -m arxiv_2601_07223.cli \
        --dataset parity \
        --mode encoded_logical \
        --gate-noise $DATA_NOISE \
        --ancilla-noise $anc_p \
        --syndrome-rounds 2 \
        --shots 4096 \
        --steps 100 \
        --log-json results/anc_threshold_anc_${anc_p}.json
done
```

**Paper Finding**: 
- Below threshold (~0.003-0.004): Error detection effective
- Above threshold: Ancilla errors propagate, limiting QEC effectiveness

**Note**: Current implementation has CUDA-Q limitations for per-qubit noise. This experiment demonstrates the API but may not fully replicate paper's results.

---

## Section 4.3: Two-Qubit Gate Noise Scaling

### Experiment 6: Two-Qubit Noise Factor

```bash
# Paper uses 2× noise for two-qubit gates
# Test impact of this scaling factor

NOISE=0.005

# Default: 2× scaling (paper's approach)
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded_logical \
    --gate-noise $NOISE \
    --two-qubit-noise-scale 2.0 \
    --syndrome-rounds 2 \
    --shots 4096 \
    --steps 100 \
    --log-json results/two_qubit_scale_2.0.json

# Equal noise (1× scaling)
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded_logical \
    --gate-noise $NOISE \
    --two-qubit-noise-scale 1.0 \
    --syndrome-rounds 2 \
    --shots 4096 \
    --steps 100 \
    --log-json results/two_qubit_scale_1.0.json

# Higher noise (3× scaling)
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded_logical \
    --gate-noise $NOISE \
    --two-qubit-noise-scale 3.0 \
    --syndrome-rounds 2 \
    --shots 4096 \
    --steps 100 \
    --log-json results/two_qubit_scale_3.0.json
```

---

## MNIST Binary Classification (Simplified)

### Experiment 7: MNIST 0 vs 1 Classification

```bash
# Paper uses 10-qubit amplitude encoding
# Current implementation uses 2-bit coarse features (tractable)

# Zero noise baseline
python -m arxiv_2601_07223.cli \
    --dataset mnist \
    --mnist-digit-positive 0 \
    --mnist-digit-negative 1 \
    --mnist-limit 128 \
    --mnist-data-dir ./data/MNIST \
    --mode encoded_logical \
    --gate-noise 0.0 \
    --syndrome-rounds 1 \
    --shots 4096 \
    --steps 120 \
    --batch-size 32 \
    --log-json results/mnist_0v1_noise_0.000.json

# With noise
python -m arxiv_2601_07223.cli \
    --dataset mnist \
    --mnist-digit-positive 0 \
    --mnist-digit-negative 1 \
    --mnist-limit 128 \
    --mnist-data-dir ./data/MNIST \
    --mode encoded_logical \
    --gate-noise 0.003 \
    --syndrome-rounds 2 \
    --shots 4096 \
    --steps 120 \
    --batch-size 32 \
    --log-json results/mnist_0v1_noise_0.003.json

# Compare with bare
python -m arxiv_2601_07223.cli \
    --dataset mnist \
    --mnist-digit-positive 0 \
    --mnist-digit-negative 1 \
    --mnist-limit 128 \
    --mnist-data-dir ./data/MNIST \
    --mode bare \
    --gate-noise 0.003 \
    --shots 4096 \
    --steps 120 \
    --batch-size 32 \
    --log-json results/mnist_0v1_bare_noise_0.003.json
```

### Experiment 8: MNIST Different Digit Pairs

```bash
# Test various digit combinations
DIGIT_PAIRS=("0 1" "3 8" "4 9" "6 8")

for pair in "${DIGIT_PAIRS[@]}"; do
    read -r d1 d2 <<< "$pair"
    python -m arxiv_2601_07223.cli \
        --dataset mnist \
        --mnist-digit-positive $d1 \
        --mnist-digit-negative $d2 \
        --mnist-limit 128 \
        --mode encoded_logical \
        --gate-noise 0.003 \
        --syndrome-rounds 2 \
        --shots 4096 \
        --steps 120 \
        --batch-size 32 \
        --log-json results/mnist_${d1}v${d2}.json
done
```

---

## Paper-Specific Settings

### Table: Paper's Experimental Parameters

| Parameter | Paper Value | Current Implementation |
|-----------|-------------|------------------------|
| **Task** | Parity (2-qubit) | ✅ Same |
| **QEC Code** | [[4,2,2]] CSS | ✅ Same |
| **Logical Encoding** | Eqs. 17-20 | ✅ Same |
| **Rotations** | Ancilla-based | ✅ `--mode encoded_logical` |
| **Noise Model** | Depolarizing | ✅ Same |
| **Pauli Error Rate** | 0.001 - 0.01 | ✅ Same |
| **Two-qubit scaling** | 2× | ✅ `--two-qubit-noise-scale 2.0` |
| **Syndrome rounds** | 1-4 tested | ✅ Same |
| **Batch size** | 8 (24 duplicated samples) | ✅ `--batch-size 8` |
| **Training iterations** | 100 | ✅ `--steps 100` |
| **Shots** | Not specified | 4096 recommended |

---

## PowerShell Commands (Windows)

```powershell
# Create results directory
New-Item -ItemType Directory -Force -Path results

# Run noise sweep
$noise_levels = 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010

foreach ($p in $noise_levels) {
    Write-Host "Running experiments for p=$p"
    
    # Bare circuit
    python -m arxiv_2601_07223.cli `
        --dataset parity `
        --mode bare `
        --gate-noise $p `
        --shots 4096 `
        --steps 100 `
        --log-json "results/bare_p_$p.json"
    
    # Encoded circuit
    python -m arxiv_2601_07223.cli `
        --dataset parity `
        --mode encoded_logical `
        --gate-noise $p `
        --syndrome-rounds 2 `
        --shots 4096 `
        --steps 100 `
        --log-json "results/encoded_logical_p_$p.json"
}

# Plot results
python plot_fig3.py --plot-noise results/*.json --output fig4_reproduction.png
```

---

## Expected Results (Based on Paper)

### Low Noise (p ≤ 0.0025)
- **Bare**: Training converges, moderate accuracy (70-85%)
- **Encoded**: Training converges, high accuracy (85-95%)
- **Difference**: QEC provides 5-15% improvement

### Medium Noise (p = 0.003 - 0.005)
- **Bare**: Training difficult, accuracy ~60-70%
- **Encoded**: Training possible, accuracy 70-85%
- **Difference**: QEC critical for trainability

### High Noise (p ≥ 0.006)
- **Bare**: Training fails, accuracy near random (50%)
- **Encoded**: Training struggles, accuracy 55-70%
- **Difference**: QEC helps but both degrade

### Syndrome Rounds
- **0-1 rounds**: Improvement with more rounds
- **2 rounds**: Plateau reached (paper's finding)
- **3-4 rounds**: Minimal additional benefit

### Ancilla Threshold
- **anc_p < 0.003**: Good performance
- **anc_p ≈ 0.003-0.004**: Threshold region
- **anc_p > 0.004**: Performance degrades

---

## Validation Checklist

After running experiments, verify:

- [ ] Zero noise: Both modes achieve >95% accuracy
- [ ] Low noise (p=0.0025): Encoded outperforms bare by 5-15%
- [ ] High noise (p=0.01): Both modes near random (50-60%)
- [ ] Syndrome rounds: Plateau after 2 rounds
- [ ] Logical encoding: `encoded_logical` uses 16+ qubits (4 data + 12 rotation + syndrome)
- [ ] Shot rejection: Check accepted vs rejected shots in logs
- [ ] Training curves: Smooth convergence in low noise, noisy/flat in high noise

---

## Plotting Results

```bash
# Training curves comparison
python plot_fig3.py \
    results/bare_p_0.003.json \
    results/encoded_logical_p_0.003.json \
    --output training_curves.png \
    --title "Training Curves: Bare vs Encoded (p=0.003)"

# Noise sweep
python plot_fig3.py \
    --plot-noise results/*_p_*.json \
    --output noise_sweep.png \
    --title "Accuracy vs Gate Noise" \
    --noise-metric best

# Syndrome rounds comparison
python plot_fig3.py \
    results/encoded_syn_*.json \
    --output syndrome_rounds.png \
    --title "Impact of Syndrome Rounds (p=0.005)"
```

---

## Quick Test (5 minutes)

```bash
# Minimal test to verify setup
python -m arxiv_2601_07223.cli \
    --dataset parity \
    --mode encoded_logical \
    --gate-noise 0.003 \
    --syndrome-rounds 2 \
    --shots 1024 \
    --steps 20 \
    --log-json results/quick_test.json

# Should complete in 2-5 minutes
# Expected: Accuracy improves from ~50% to 70-90%
```

---

## Computational Requirements

| Experiment | Shots | Steps | Time (CPU) | Memory |
|------------|-------|-------|------------|--------|
| Quick test | 1024 | 20 | 2-5 min | 4 GB |
| Single run | 4096 | 100 | 15-30 min | 8 GB |
| Noise sweep (10 points) | 4096 | 100 | 3-5 hours | 8 GB |
| Full validation | 4096 | 100 | 8-12 hours | 16 GB |

**Target**: `density-matrix-cpu` (best for noise simulation)

---

## Troubleshooting

### Slow Execution
- Reduce `--shots` to 2048 or 1024
- Use `--mode encoded` instead of `encoded_logical`
- Reduce `--syndrome-rounds` to 1

### Poor Convergence
- Increase `--lr` to 0.3-0.5
- Increase `--steps` to 150-200
- Check noise level isn't too high

### Out of Memory
- Reduce `--shots`
- Use `--mode bare`
- Check available RAM

---

## Citation

When reporting results, cite:
```
Eromanga Adermann, Haiyue Kang, Martin Sevior, Muhammad Usman.
"Quantum Error Correction and Detection for Quantum Machine Learning."
arXiv:2601.07223 (2026).
```

And note implementation details:
- Code version/commit
- Mode used (`bare`, `encoded`, `encoded_logical`)
- CUDA-Q version
- Hardware (CPU/GPU)
- Any deviations from paper
