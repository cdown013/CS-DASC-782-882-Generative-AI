# Fashion-MNIST VAE Write-up

## Citation

David Foster, *Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play*, 2nd edition, O’Reilly Media, 2023, Chapter 3 (Variational Autoencoders). Companion notebook: [vae_fashion.ipynb](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/03_vae/02_vae_fashion/vae_fashion.ipynb).

## Setup

- TensorFlow 2.21.0 / Keras 3.15.1, CPU only, seed **42**
- Fashion-MNIST padded to 32×32×1, Adam `5e-4`, **5 epochs**, batch size 100, `beta=500`
- Latent-dimension ablation: **2 / 8 / 32** (fresh model each run)
- Total wall time ≈ **337 s** (~5.6 min) for all three runs

## ELBO and the logged losses

A VAE maximizes an evidence lower bound (ELBO) on \(\log p(x)\). With a Bernoulli decoder on pixels in \([0,1]\), the practical training loss is the negative of that bound and splits into two terms:

1. **Reconstruction:** \(\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]\), implemented as pixel-wise **binary cross-entropy**, summed over the spatial map and scaled by \(\beta=500\) (Foster Ch.3). Large \(\beta\) makes the optimizer care more about rebuilding \(x\) than about matching the prior exactly.
2. **KL divergence:** \(D_{KL}(q_\phi(z|x)\,\|\,p(z))\) with \(p(z)=\mathcal{N}(0,I)\). This regularizes each approximate posterior toward the standard normal so the latent space stays continuous and prior samples decode sensibly.

The notebook and `history_dim{N}.csv` files log three scalars every epoch: `loss` (= reconstruction + KL), `reconstruction_loss`, and `kl_loss` (train and validation).

## Ablation results (epoch 5, train)

| Latent dim | Loss | Reconstruction | KL | Train seconds |
|-----------:|-----:|---------------:|---:|--------------:|
| 2 | 129957.9 | 129932.0 | 25.9 | 106.9 |
| 8 | 112418.3 | 112336.0 | 82.3 | 109.2 |
| 32 | 105810.3 | 105634.6 | 175.7 | 114.2 |

Validation tracks training closely (e.g. dim-2 val loss 130784; dim-32 val loss 106204), so the runs are not badly overfit after five epochs.

**What the numbers say**

- Reconstruction dominates the numeric objective because of \(\beta=500\); KL is two to four orders of magnitude smaller but still informative.
- Widening the bottleneck from 2 → 32 cuts reconstruction loss by about **19%** while KL grows about **7×**. Extra latent axes let \(q(z|x)\) carry more structure, which improves pixel fidelity and moves the aggregate posterior farther from \(\mathcal{N}(0,I)\).
- KL never collapses: final KL ≈ **26 / 82 / 176** for dims 2 / 8 / 32. The encoder is not ignoring \(z\).

## Qualitative failure modes

Observed in `artifacts/reconstructions.png`, `prior_samples*.png`, and `latent_scatter*.png`:

- **Dim 2 (blurry reconstructions).** Silhouettes are right (boots vs trousers vs tops), but logos, checks, and edges smear into gray blobs. Capacity is too small to preserve texture.
- **Dim 2 (class overlap in the plane).** Trousers form a clear left lobe; sandals / sneakers / ankle boots pile on the right; shirts, coats, and pullovers heavily overlap in the center. Bags sit in a small mixed region. The 2D map is organized but not linearly separable by class.
- **Prior samples are ghostly.** Draws from \(\mathcal{N}(0,I)\) look like clothing, yet many are soft blends (boot-shirt hybrids, washed-out sleeves) - expected when reconstruction is weighted far above KL and the decoder averages modes.
- **Dim 8 / 32 improve fidelity.** Edges sharpen and some internal structure returns; prior grids look more item-like. The price is a less readable 2D PCA projection of an 8- or 32-D code: structure still exists, but it is no longer a single clean plane.
- **Not observed:** posterior collapse (KL → 0) or divergent train/val curves within five epochs.

## Takeaway

Under Foster’s \(\beta=500\) recipe, the VAE trades a soft, continuous latent prior for reconstruction quality. Two latent dimensions are enough to organize major shape families and sample recognizable clothes; eight and thirty-two dimensions buy sharper reconstructions at the cost of a denser, higher-KL latent code and a scatter plot that must be projected to be viewed.
