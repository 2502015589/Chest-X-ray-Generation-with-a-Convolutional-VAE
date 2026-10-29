# Chest X-ray Generation with a Convolutional VAE

A PyTorch variational autoencoder for unconditional generation of **128 x 128 grayscale chest X-rays**. 

## Files

```text
README.md
requirements.txt
as1.ipynb
outputs/
  loss_history.json
  loss_curves.png
  final_samples.png
  samples_epoch10.png
  samples_epoch20.png
  samples_epoch30.png
  samples_epoch40.png
  samples_epoch50.png
  samples_epoch60.png
  samples_epoch70.png
  samples_epoch80.png
  metrics_128.json
```

`as1.ipynb` at the repository root is the project notebook. Run it from the repository root so its relative data and output paths resolve correctly. Saved experiment artifacts are in `outputs/`. 

## Dataset

Download [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) , and place it at the repository root:

```text
chest_xray/
  train/NORMAL/
  train/PNEUMONIA/
  val/NORMAL/
  val/PNEUMONIA/
  test/NORMAL/
  test/PNEUMONIA/
```

Local split sizes: train 5,216; validation 16; test 624. The model is trained on the training set. The final-epoch checkpoint is used for generation and evaluated against the test set using FID and Inception Score.

## Setup

Environment:

```
Python 3.13.2
PyTorch 2.6.0 with CUDA 12.4
NVIDIA GeForce RTX 4050 Laptop GPU
```

Dependencies: 

```bash
python -m pip install -r requirements.txt
```

Open `as1.ipynb` and execute the cells in order. Use the project root as the working directory.


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
| Epochs | 80 |
| KL weight | 1.0 |
| Training / evaluation seed | 42 / 342 |

The objective is pixel-summed BCE plus KL divergence, averaged over images. The data loader uses `drop_last=True`, omitting the final incomplete batch of 32 images each epoch. Model weights are saved to `outputs/vae.pt` after training.

## Existing results

| Metric | Value |
|---|---:|
| Final training total loss | 10250.8882 |
| Final reconstruction loss | 10206.5100 |
| Final KL | 44.3783 |
| Ignite-default FID (lower is better) | 0.152624 |
| Inception Score | 1.670664 |
| Real / generated evaluation count | 624 / 624 |

Losses are in nats per image. Values are taken from the current saved JSON files, not earlier screenshots.

![Training loss curves](outputs/loss_curves.png)

![Generated samples](outputs/final_samples.png)

Generated samples from fixed latent vectors are saved every 10 epochs through epoch 80. The experiment configuration is recorded in `outputs/config.json`.

## Training interpretation

The training loss decreased from 10920.29 to 10250.89 over 80 epochs. It dropped rapidly during the early stages and gradually stabilized as training progressed. Generated images capture coarse thoracic structure but remain blurred and lack fine anatomical detail. Lower training loss alone does not establish better generation quality.

## Evaluation

The model was evaluated using 624 real test images and 624 generated images. Both real and generated images were converted to RGB and resized to 299 × 299 using bilinear interpolation. FID and Inception Score were computed using PyTorch-Ignite, yielding 0.1526 and 1.6707, respectively.

