# arxiv-quantum

Utilities and experiments replicating aspects of "Quantum Error Correction and Detection
for Quantum Machine Learning" (arXiv:2601.07223) with CUDA-Q.

## 📚 Documentation Index

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **[README.md](README.md)** (this file) | Setup, usage, quick start | 5 min |
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** | Commands, status at a glance | 3 min |
| **[SUMMARY.md](SUMMARY.md)** | Executive overview, decisions | 10 min |
| **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** | Detailed verification report | 20 min |
| **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** | Technical specifications | 30 min |
| **[ROADMAP.md](ROADMAP.md)** | Week-by-week schedule | 15 min |
| **[PROGRESS.md](PROGRESS.md)** | Task tracking, metrics | 10 min |

**Quick Navigation**:
- 🚀 **New user?** → Start with this README, then [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- 🔍 **Need to verify alignment?** → [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
- 🛠️ **Want to contribute?** → [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) + [ROADMAP.md](ROADMAP.md)
- 📊 **Track progress?** → [PROGRESS.md](PROGRESS.md)
- 🎯 **Make decisions?** → [SUMMARY.md](SUMMARY.md)

## Implementation Status

**For detailed verification report, see [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)**

### ✅ Implemented (Section 4: Error Detection)
- [[4,2,2]] Calderbank-Shor-Steane stabilizer code
- Syndrome extraction and shot rejection logic
- Depolarizing noise model (single and two-qubit gates)
- 2-qubit parity classification task
- Parameter-shift gradient descent training
- Logical rotation encoding (paper-correct implementation)

### ⚠️ Partial Implementation
- **Rotation gates**: Both simplified and paper-correct implementations provided
  - `--mode encoded`: Rotations applied directly to physical qubits (NOT fault-tolerant, but computationally efficient)
  - `--mode encoded_logical`: Full ancilla-based logical rotations per paper Section 4.2 (✅ paper-correct)
- **Multi-layer support**: Bare circuits support arbitrary depth; encoded circuits at 2 layers
- **Dataset**: 2-bit coarse features from MNIST vs. paper's full amplitude encoding

### ❌ Not Implemented
- Section 2: Full QEC resource analysis with Azure quantum resource estimator
- Section 3: Partial QEC protocol (error-corrected Clifford gates, raw T gates)
- Deep circuits: Code implements single-layer circuits; paper uses 75-100 layers for MNIST
- Ancilla-specific noise rates (limited by CUDA-Q API; parameter added but not fully functional)

## Circuit Modes

- **`bare`**: Unencoded 2-qubit circuit (no error detection)
- **`encoded`**: [[4,2,2]] code with simplified direct rotations (⚡ fast, ❌ NOT paper-correct)
- **`encoded_logical`**: [[4,2,2]] code with full logical rotation encoding (✅ paper-correct, slower)

## Setup

```bash
python -m venv cudaq-env
source cudaq-env/bin/activate
pip install -r requirements.txt
```

Make sure CUDA-Q is properly installed for your platform (see NVIDIA’s docs) and
select a simulator that supports noise, e.g. `density-matrix-cpu`.

## Running Experiments

- **Parity baseline**

  ```bash
  python test_cudaq.py --dataset parity --mode encoded --syndrome-rounds 1 --shots 2048
  ```

  Switch to `--mode bare` to compare the unencoded circuit with identical
  optimizer settings.

- **MNIST binary classification**

  ```bash
  python test_cudaq.py --dataset mnist --mnist-digit-positive 0 --mnist-digit-negative 1 \\
      --mnist-limit 128 --mode encoded --syndrome-rounds 1 --shots 4096 --steps 120 --batch-size 32
  ```

  The loader downloads MNIST into `./data` (configurable via `--mnist-data-dir`),
  extracts two coarse binary features (top-vs-bottom and left-vs-right intensity),
  and trains the same single-parameter VQC so you can compare encoded vs. bare
  performance on a realistic dataset.

Refer to `docs/cudaq-notes.md` for kernel-language constraints and noise-modeling
tips relevant to these scripts.

## Reproducing Figure 3

1. Record training curves for both circuit variants (encoded vs. bare) using the CLI logging flag:

   ```bash
   python -m arxiv_2601_07223.cli --dataset mnist --mode encoded \
       --mnist-digit-positive 0 --mnist-digit-negative 1 \
       --mnist-limit 128 --mnist-data-dir ./data/MNIST \
       --shots 4096 --steps 120 --batch-size 32 \
       --log-json logs/mnist_encoded.json

   python -m arxiv_2601_07223.cli --dataset mnist --mode bare \
       --mnist-digit-positive 0 --mnist-digit-negative 1 \
       --mnist-limit 128 --mnist-data-dir ./data/MNIST \
       --shots 4096 --steps 120 --batch-size 32 \
       --log-json logs/mnist_bare.json
   ```

   The `--log-json` argument dumps per-step loss/accuracy so the results can be plotted even on
   machines without interactive CUDA-Q sessions.

2. Plot the runs side-by-side (matching arXiv Figure 3) with:

   ```bash
   python plot_fig3.py logs/mnist_encoded.json logs/mnist_bare.json --output fig3.png
   ```

   The script overlays accuracy vs. training step for each log, helping you confirm the encoded
   advantage reported in the paper.

3. To reproduce the paper’s noise sweep, capture a log for each gate-noise value you care about
   (do this for *both* encoded and bare circuits):

   ```bash
   python -m arxiv_2601_07223.cli --dataset mnist --mode encoded \
       --mnist-digit-positive 0 --mnist-digit-negative 1 \
       --mnist-limit 128 --mnist-data-dir ./data/MNIST \
       --shots 4096 --steps 120 --batch-size 32 \
       --gate-noise 0.00 --log-json logs/encoded_noise_000.json

   python -m arxiv_2601_07223.cli --dataset mnist --mode encoded \
       --mnist-digit-positive 0 --mnist-digit-negative 1 \
       --mnist-limit 128 --mnist-data-dir ./data/MNIST \
       --shots 4096 --steps 120 --batch-size 32 \
       --gate-noise 0.02 --log-json logs/encoded_noise_002.json

   # ...repeat for additional noise values plus the corresponding bare runs
   ```

   Then convert those logs into the Figure 3 noise plot:

   ```bash
   python plot_fig3.py --plot-noise logs/encoded_noise_*.json logs/bare_noise_*.json \
       --output fig3_noise.png --title "Encoded vs. bare accuracy vs. gate noise"
   ```

   By default the noise plot uses the best recorded accuracy from each log; switch to `--noise-metric last`
   if you prefer final-step accuracy or `--noise-metric mean --noise-mean-window 10` to average the last
   few iterations.

## Documentation

### Quick Links
- **[SUMMARY.md](SUMMARY.md)** - Executive overview and next steps
- **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** - Detailed verification report
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Technical plan for missing features
- **[ROADMAP.md](ROADMAP.md)** - Week-by-week implementation schedule

### What's Implemented
The current codebase implements **Section 4** of the paper (Error Detection with [[4,2,2]] code) with high fidelity, including the paper-correct logical rotation encoding using ancilla qubits.

### What's Missing
The paper's **core contributions** (Sections 2-3: Resource Analysis and Partial QEC) are not yet implemented. See IMPLEMENTATION_PLAN.md for a comprehensive 15-week plan to address all missing features.

### Critical Path
For paper validation, focus on:
1. **Phase 1**: Deep circuit support (weeks 1-3)
2. **Phase 2**: Partial QEC protocol (weeks 4-7)

This validates the paper's main innovation: training quantum circuits at noise levels where traditional approaches fail.

## Contributing

See ROADMAP.md for development phases and task tracking. Key areas for contribution:
- Clifford+T gate decomposition (Phase 2)
- Multi-layer encoded circuits (Phase 1)
- Ancilla noise modeling (Phase 3)
- Azure Quantum integration (Phase 5)

## Citation

If you use this code, please cite the original paper:
```
Eromanga Adermann, Haiyue Kang, Martin Sevior, Muhammad Usman. 
"Quantum Error Correction and Detection for Quantum Machine Learning." 
arXiv:2601.07223 (2026).
```

## License

[Specify license - e.g., MIT, Apache 2.0]
