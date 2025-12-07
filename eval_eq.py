# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

Ref:
    https://github.com/sihyun-yu/REPA/blob/main/generate.py
"""

import argparse
import gc
import json
import os
import torch.nn.functional as F

from dictdot import dictdot
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist

from models.autoencoder import vae_models
from utils import load_encoders

from torchvision.datasets import VisionDataset
from glob import glob
import torchvision.transforms.v2 as transforms
from torch.utils.data.distributed import DistributedSampler
from pit.evaluations.psnr import get_psnr
from torch.utils.data import DataLoader

class SimpleDataset(VisionDataset):
    def __init__(self, root: str, image_size):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        if root.endswith(".txt"):
            with open(root) as f:
                lines = f.readlines()
            self.fpaths = [line.strip("\n") for line in lines]
        else:
            self.fpaths = sorted(glob(root + "/**/*.JPEG", recursive=True))
            self.fpaths += sorted(glob(root + "/**/*.jpg", recursive=True))
            self.fpaths += sorted(glob(root + "/**/*.png", recursive=True))

        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "img": img,
            "fpath": fpath,
        }


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    seed = (args.global_seed + rank) * dist.get_world_size()//2
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    dataset = SimpleDataset("/workspace/cogview_dev/xutd/xu/datasets/ILSVRC/Data/CLS-LOC/val", 256)
    sampler = DistributedSampler(dataset, drop_last=True)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler)

    if args.exp_path is None or args.train_steps is None:
        if rank == 0:
            print("The `exp_path` or `train_steps` is not provided, setting `exp_path` and `train_steps` to default values.")
        args.exp_path = "pretrained/sit-xl-dinov2-b-enc8-repae-sdvae-0.5-1.5-400k"
        args.train_steps = 400_000

    with open(os.path.join(args.exp_path, "args.json"), "r") as f:
        config = dictdot(json.load(f))

    # Load model:
    if config.vae == "f8d4" or config.vae == "f8d4flow":
        latent_size = config.resolution // 8
        in_channels = 4
    elif config.vae == "f16d32" or config.vae == "f16d32flow":
        latent_size = config.resolution // 16
        in_channels = 32
    else:
        raise NotImplementedError()

    block_kwargs = {"fused_attn": config.fused_attn, "qk_norm": config.qk_norm}

    train_step_str = str(args.train_steps).zfill(7)
    state_dict = torch.load(
        os.path.join(args.exp_path, "checkpoints", train_step_str +'.pt'),
        map_location=f"cuda:{device}",
    )

    # Load the VAE and latent stats
    vae = vae_models[config.vae]().to(device)
    if "vae" in state_dict:
        # REPA-E checkpoints, VAE is in the checkpoint
        vae_state_dict = state_dict['vae']

        latents_scale = state_dict["ema"]["bn.running_var"].rsqrt().view(1, in_channels, 1, 1).to(device)
        latents_bias = state_dict["ema"]["bn.running_mean"].view(1, in_channels, 1, 1).to(device)
    else:
        # LDM-training-only checkpoints, VAE checkpoint should be in the config
        vae_state_dict = torch.load(config.vae_ckpt, map_location=f"cuda:{device}")

        latents_stats = torch.load(
            config.vae_ckpt.replace(".pt", "-latents-stats.pt"),
            map_location=f"cuda:{device}"
        )
        latents_scale = latents_stats["latents_scale"].to(device)
        latents_bias = latents_stats["latents_bias"].to(device)
        del latents_stats

    vae.load_state_dict(vae_state_dict)
    vae.eval()

    gc.collect()
    torch.cuda.empty_cache()
    all_psnr = [[] for _ in range(world_size)]
    all_kl = [[] for _ in range(world_size)]

    total_num = 0

    for bi, batch in enumerate(loader):
        img = batch["img"].to(device)
        z = vae.encode(img, sample=True)
        kl = torch.sqrt(torch.sum(z ** 2, dim=(1,2,3)))
        img_hat = vae.decode(z).sample
        pred_psnr = get_psnr(img, img_hat, zero_mean=True)

        gathered_psnr = [torch.zeros_like(pred_psnr) for _ in range(world_size)]
        gathered_kl = [torch.zeros_like(kl) for _ in range(world_size)]

        torch.distributed.all_gather(gathered_psnr, pred_psnr)
        torch.distributed.all_gather(gathered_kl, kl)

        for j in range(world_size):
            all_psnr[j].append(gathered_psnr[j].detach().cpu())
            all_kl[j].append(gathered_kl[j].detach().cpu())

        total_num += world_size * img.shape[0]

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        print("done.")
        for j in range(world_size):
            all_psnr[j] = torch.cat(all_psnr[j], dim=0).numpy()
            all_kl[j] = torch.cat(all_kl[j], dim=0).numpy()

        all_psnr_reorg = []
        all_kl_reorg = []

        for j in range(total_num):
            all_psnr_reorg.append(all_psnr[j % world_size][j // world_size])
            all_kl_reorg.append(all_kl[j % world_size][j // world_size])

        all_psnr = np.vstack(all_psnr_reorg)
        all_kl = np.vstack(all_kl_reorg)

        print(f"PSNR: {np.mean(all_psnr):.4f} (±{np.std(all_psnr):.4f})")
        print(f"KL eq: {np.mean(all_kl):.4f} (±{np.std(all_kl):.4f})")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed params
    parser.add_argument("--global-seed", type=int, default=0)

    # precision params
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving params
    parser.add_argument("--sample-dir", type=str, default="samples")

    # ckpt params
    parser.add_argument("--exp-path", type=str, default=None, help="Path to the specific experiment directory.")
    parser.add_argument("--train-steps", type=str, default=None, help="The checkpoint of the model to sample from.")

    # number of samples
    parser.add_argument("--pproc-batch-size", type=int, default=256)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False,
                        help="Use Heun's method for ODE sampling.")
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    args = parser.parse_args()
    main(args)
