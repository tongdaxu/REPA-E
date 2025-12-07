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
import json
import os
import torch.nn.functional as F

from dictdot import dictdot
import torch
import torch.nn as nn

from models.autoencoder import vae_models

from torch.utils.data import DataLoader
from accelerate import Accelerator
from foundation_models import aux_foundation_model
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

from dataset import CustomINH5Dataset
from utils import preprocess_imgs_vae
import copy
from pathlib import Path

from train_repae_flow import create_logger
from metrics import AlignmentMetrics

class VFModel(nn.Module):
    def __init__(self, embed_dim, vf_feature_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.vf_feature_dim = vf_feature_dim
        self.linear_proj = torch.nn.Conv2d(embed_dim, vf_feature_dim, kernel_size=1, bias=False)
        self.foundation_model = aux_foundation_model("dinov2")
        for param in self.linear_proj.parameters():
            param.requires_grad_(True)
        for param in self.foundation_model.parameters():
            param.requires_grad_(False)

    def forward(self, x, z):
        b, _, _, _ = x.shape
        with torch.no_grad():
            aux_feature = self.foundation_model(x)
        zp = F.interpolate(z, size=(aux_feature.shape[2], aux_feature.shape[3]), mode="bilinear")
        zp = torch.repeat_interleave(zp, self.vf_feature_dim // self.embed_dim, dim=1)
        aux_feature = aux_feature.reshape(b, -1)
        zp = zp.reshape(b, -1)
        n_ax = F.normalize(aux_feature, dim=-1)
        n_zp = F.normalize(zp, dim=-1)
        score = AlignmentMetrics.cknna(n_ax, n_zp, topk=10)
        loss = torch.tensor([score], device=x.device)
        return loss

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )
    assert(args.gradient_accumulation_steps == 1)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # set up the logger and checkpoint dirs
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Setup data
    train_dataset = CustomINH5Dataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    if args.exp_path is None or args.train_steps is None:
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
        map_location=device,
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
        vae_state_dict = torch.load(config.vae_ckpt, map_location=device)

        latents_stats = torch.load(
            config.vae_ckpt.replace(".pt", "-latents-stats.pt"),
            map_location=device
        )
        latents_scale = latents_stats["latents_scale"].to(device)
        latents_bias = latents_stats["latents_bias"].to(device)
        del latents_stats

    vae.load_state_dict(vae_state_dict)

    model = VFModel(in_channels)

    # Define the optimizers for SiT, VAE, and VAE loss function separately
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    model, vae, optimizer, train_dataloader = accelerator.prepare(
        model, vae, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="gradient-pass-through",
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )

    global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(args.epochs):
        model.train()
        vae.eval()
        for raw_image, y in train_dataloader:
            raw_image = raw_image.to(device)
            processed_image = preprocess_imgs_vae(raw_image)
            with torch.no_grad():
                z = accelerator.unwrap_model(vae).encode(processed_image, sample=True)
            vf_loss = model(processed_image, z)
            # enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # Prepare the logs based on the current step
                logs = {
                    "vf_loss": accelerator.gather(vf_loss).mean().detach().item(),
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

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
    
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--epochs", type=int, default=1)

    args = parser.parse_args()
    main(args)
