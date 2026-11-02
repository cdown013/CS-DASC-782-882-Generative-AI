# Fashion-MNIST VAE (Foster Ch.3 assignment)

Keras VAE on Fashion-MNIST with logged ELBO terms, a `{2, 8, 32}` latent-dim ablation, prior samples, and class-colored latent scatter.

Based on David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023, Chapter 3.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pins: TensorFlow **2.21.0**, Keras **3.x**, Python 3.13 (macOS arm64 CPU). No `tensorflow-metal`.

## Seed and hyperparameters

| Setting | Value |
|--------|------:|
| Seed | 42 |
| Beta (recon weight) | 500 |
| Epochs | 5 |
| Batch size | 100 |
| Learning rate (Adam) | 5e-4 |
| Image size | 32×32×1 |
| Latent dims | 2, 8, 32 |

## Run order

1. **Full ablation (writes artifacts + models)**  
   `python run_experiments.py --latent-dims 2 8 32`  
   Expected wall time on this CPU laptop: ~5–6 minutes total.

2. **Notebook (loads trained models + histories)**  
   `jupyter notebook vae_fashion_assignment.ipynb`  
   or  
   `jupyter nbconvert --to notebook --execute vae_fashion_assignment.ipynb --output vae_fashion_assignment_executed.ipynb`

3. **Write-up**: see `WRITEUP.md` (numbers taken from `artifacts/metrics.json`).

## Deliverables

- `vae_fashion.py`: data, Sampling, encoder/decoder, VAE, plots
- `run_experiments.py`: headless ablation
- `vae_fashion_assignment.ipynb`: graded narrative notebook
- `artifacts/`: `metrics.json`, `history_dim{N}.csv`, PNG figures
- `models/`: `encoder_dim{N}.keras`, `decoder_dim{N}.keras`, `vae_dim{N}.keras`
- `WRITEUP.md`: interpretation grounded in real metrics

## Expected artifacts

- `loss_curves.png` (+ per-dim variants)
- `ablation_curves.png`
- `reconstructions.png`
- `prior_samples.png` (+ per-dim)
- `latent_scatter.png` (+ per-dim; PCA noted for dim > 2)
- `metrics.json` with `loss`, `reconstruction_loss`, `kl_loss` for each dim

## FLWR drafting note

`artifacts/writeup_context.md` and `artifacts/writeup_prompt.md` were generated from real `metrics.json` for optional drafting with `flwr run flwr-bloom`. FLWR stalled / produced an incomplete draft with invented numbers, so it was skipped for the final submission. `WRITEUP.md` was written directly from `metrics.json`, history CSVs, and the figure observations.
