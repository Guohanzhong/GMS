# Gaussian Mixture Solvers for Diffusion Models

This repo is the official code for the paper [Gaussian Mixture Solvers for Diffusion Models](https://openreview.net/forum?id=0NuseeBuB4) (NeurIPS 2023 Poster).

## Requirements
- Python 3.8
- Packages
    Upgrade pip for installing latest tensorboard
    ```
    pip install -U pip setuptools
    pip install -r requirements.txt
    ```
- Download precalculated statistic for dataset:

    [cifar10.train.npz](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing)

    Create folder `stats` for `cifar10.train.npz`.
    ```
    stats
    └── cifar10.train.npz
    ```

## Train From Scratch
- Take CIFAR10 for example, training noise network 
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python main.py --flagfile=./config/CIFAR10_iddpm.txt --train  --noise_order 1 --parallel --logdir='dir' --noise_schedule linear/cosine --total_steps total_steps --mode simple/complex --pretrained_dir ''pre-trained dir > ./train_logs/train_cos_3.log 2>&1 &
    ```
- Take CIFAR10 for example, training higher-order noise network 
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python main.py --flagfile=./config/CIFAR10_iddpm.txt --train  --noise_order 3 --parallel --logdir='dir' --noise_schedule linear/cosine --total_steps total_steps --mode simple/complex --pretrained_dir ''pre-trained dir > train.log 2>&1 &
    ```
-  Difference would be the choose of noise_order, if set noise_order $\ge 2$, the pretrained dir is required


## Evaluate

- Start evaluation on CIFAR10
    ```
    CUDA_VISIBLE_DEVICES nohup python sample.py --flagfile=./config/CIFAR10_iddpm.txt --parallel --batch_size bzs --mode simple/complex --sample_type ddpm/analyticdpm/gmddpm --sample_steps K --num_images 50000 > evaluation.log 2>&1 &
    ```
* `bzs` is the batch size for sampling.




