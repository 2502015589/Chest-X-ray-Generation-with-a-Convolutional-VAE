# Chest X-ray Generation with a Convolutional VAE

A PyTorch variational autoencoder for unconditional generation of **128 x 128 grayscale chest X-rays**. NORMAL and PNEUMONIA images are pooled without conditioning on labels.

## Files

```text
README.md
requirements.txt
.gitignore
notebooks/
  as1.ipynb
results/
  config.json
  loss_history.json
  loss_curves.png
  final_samples.png
  samples_epoch10.png
  samples_epoch20.png
  samples_epoch30.png
  metrics_128.json
  provenance.json
```

`notebooks/as1.ipynb` is the upload-ready copy of the current root notebook. The test-transform definition was moved before construction of the test dataset so the evaluation cells can execute in order. Training and evaluation logic are otherwise unchanged. The root notebook is preserved locally. This package excludes old decoder experiments and report files.

## Dataset

Download [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) separately, following its applicable usage terms. Place it at the repository root:

```text
chest_xray/
  train/NORMAL/
  train/PNEUMONIA/
  val/NORMAL/
  val/PNEUMONIA/
  test/NORMAL/
  test/PNEUMONIA/
```

Local split sizes: train 5,216; validation 16; test 624. This original notebook trains on the training split and evaluates generation on the test split. It does not use validation-based checkpoint selection.

## Installation and running

Use a virtual environment, then:

```bash
python -m pip install -r requirements.txt
```

Open `notebooks/as1.ipynb` in VS Code or Jupyter and select that environment. Python 3.13.2 is recorded in the source notebook. For a browser interface, optionally install JupyterLab with `python -m pip install jupyterlab` and launch `python -m jupyterlab`.

**Use the repository root as the kernel working directory**, because dataset and output paths are relative to it. If the kernel starts inside `notebooks/`, run this once in a temporary cell before the existing cells:

```python
from pathlib import Path
import os
if Path.cwd().name == "notebooks":
    os.chdir(Path.cwd().parent)
print(Path.cwd())
```

Run the cells in order to train and evaluate. Running all cells starts a fresh 30-epoch training run and writes to `outputs/`; back up any existing outputs first. **No training or evaluation was rerun to prepare these repository files.**

For CUDA, use a matching PyTorch/torchvision CUDA build from the [official installation selector](https://pytorch.org/get-started/locally/). The dependency pins record local package versions but do not specify a CUDA build. Inception weights require network access on first metric evaluation. If DataLoader workers hang on Windows/Jupyter, try `num_workers=0` instead of the original setting of 2.

## Model and hyperparameters

Five encoder blocks use stride-2 convolutions with channels 32, 64, 128, 256, 512, BatchNorm and LeakyReLU. Linear heads predict a diagonal Gaussian mean and log-variance. Reparameterization produces latent samples. The decoder uses transposed convolutions with BatchNorm/ReLU and a final Sigmoid.

| Setting | Value |
|---|---|
| Image size | 128 x 128, grayscale |
| Preprocessing | Resize preserving aspect ratio, CenterCrop, ToTensor [0,1] |
| Latent dimension | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 30 |
| KL weight | 1.0 |
| Training / evaluation seed | 42 / 342 |

The objective is pixel-summed BCE plus KL divergence, averaged over images. The original loader uses `drop_last=True`, so 32 shuffled training images are omitted each epoch. Final-epoch weights are saved to `outputs/vae.pt` and are not included in the upload package.

## Existing results

| Metric | Value |
|---|---:|
| Final training total loss | 10271.2665 |
| Final reconstruction loss | 10224.7753 |
| Final KL | 46.4913 |
| Ignite-default FID (lower is better) | 0.100035 |
| Inception Score | 1.414462 |
| Real / generated evaluation count | 624 / 624 |

Losses are in nats per image. Values are taken from the current saved JSON files, not earlier screenshots.

![Training loss curves](results/loss_curves.png)

![Generated samples](results/final_samples.png)

Fixed-latent samples from epochs 10, 20 and 30 are also included. `config.json` records settings extracted from the source notebook; `provenance.json` records primary source paths, checksums and the local packaging environment.

## Evaluation and limitations

Evaluation follows the [assignment's Ignite tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/). The model generates 624 independent samples from the standard-normal latent prior. Real test images use the original Resize + CenterCrop preprocessing. Both streams are converted to RGB and resized with PIL bilinear interpolation to 299 x 299. FID uses both streams; Inception Score uses only generated images.

These scores use **Ignite's default 1000-dimensional ImageNet Inception output**, rather than pool3-2048 FID. Do not compare them directly against other FID implementations. The sample size is limited, and ImageNet features do not establish anatomical or diagnostic accuracy. Generated images capture coarse structure but remain blurred. Fixed seeds do not guarantee identical results across hardware and software environments.

## GitHub upload

Commit `README.md`, `requirements.txt`, `.gitignore`, `notebooks/`, and `results/`. Do not include the dataset, weights, virtual environments or older experiments. Review the course policy before publishing assignment solutions. No license is assumed or assigned by this packaging step.
