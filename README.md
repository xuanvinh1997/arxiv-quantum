# arxiv-quantum

Utilities and experiments replicating the “Quantum Error Correction and Detection
for Quantum Machine Learning” study (arXiv:2601.07223) with CUDA-Q.

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
