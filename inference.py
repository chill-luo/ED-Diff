import io
import os
import time
import datetime
import argparse
import numpy as np
import blobfile as bf
import tifffile as tiff

import torch
from torch.utils.data import DataLoader

from dataset import TifDataset
from sampling import eddiff_sample
from utils import save_args_to_yaml
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    print("Creating models...")

    with bf.BlobFile(args.model_path, "rb") as f:
        model_file = f.read()
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(io.BytesIO(model_file), map_location="cpu"))
    model = model.cuda()
    model.eval()
    args.total_steps = diffusion.num_timesteps

    save_dir = args.save_dir
    os.makedirs(os.path.join(save_dir, "output"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "vis"), exist_ok=True)
    save_args_to_yaml(args, os.path.join(save_dir, "args.yaml"))
    print(f"Target directory: {save_dir}.")

    elapsed_time_list = []
    print("Start sampling...")
    name = os.path.basename(args.data_pth).replace(".tif", "")
    stack = tiff.imread(args.data_pth)
    if stack.ndim == 2:
        stack = stack[None, ...]
    if args.gt_pth is not None:
        gt_stack = tiff.imread(args.gt_pth)
        if gt_stack.ndim == 2:
            gt_stack = gt_stack[None, ...]
    n, h, w = stack.shape
    for ind in range(n):
        noisy = stack[ind].astype(np.float32)
        h_new = min(h - h % 32, args.max_size)
        w_new = min(w - w % 32, args.max_size)
        h_start = (h - h_new) // 2
        w_start = (w - w_new) // 2

        p1 = int(noisy[h_start:h_start+h_new, w_start:w_start+w_new].min())
        p2 = int(noisy[h_start:h_start+h_new, w_start:w_start+w_new].max())
        noisy = noisy[h_start:h_start+h_new, w_start:w_start+w_new]
        if args.gt_pth is not None:
            gt = gt_stack[ind, h_start:h_start+h_new, w_start:w_start+w_new]

        noisy = 2 * (noisy - p1) / (p2 - p1) - 1
        noisy = torch.tensor(noisy[None, None, :, :]).cuda()

        start_time = time.time()
        all_samples, _, _, _, _, _, _ = eddiff_sample(
            model,
            diffusion,
            noisy,
            p1,
            p2,
            args
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_list.append(elapsed_time)

        denoised = ((all_samples[-1].numpy() + 1) / 2 * (p2 - p1) + p1).clip(0, 65525).astype(np.uint16)
        noisy = ((noisy.cpu().numpy() + 1) / 2 * (p2 - p1) + p1).clip(0, 65525).astype(np.uint16)
        tiff.imwrite(os.path.join(save_dir, "output", f"{str(ind+1).zfill(5)}_{name}.tif"), denoised[0, 0, :, :].astype(np.uint16))

        def u16tou8(x):
            x = x.astype(np.float32)
            x = (x - x.min()) / (x.max() - x.min()) * 255
            return x.clip(0, 255).astype(np.uint8)

        denoised_vis = u16tou8(denoised[0, 0, :, :])
        noisy_vis = u16tou8(noisy[0, 0, :, :])
        if args.gt_pth is not None:
            gt_vis = u16tou8(gt.astype(np.uint16))
            tiff.imwrite(os.path.join(save_dir, f"vis_{str(ind+1).zfill(5)}_{name}.tif"), np.hstack((noisy_vis, denoised_vis, gt_vis)).astype(np.uint8))
        else:
            tiff.imwrite(os.path.join(save_dir, f"vis_{str(ind+1).zfill(5)}_{name}.tif"), np.hstack((noisy_vis, denoised_vis)).astype(np.uint8))

        left_time = datetime.timedelta(seconds=int(np.mean(elapsed_time_list) * (n - ind - 1)))
        print(f"Created {name}-{ind+1} within {elapsed_time:.2f}s. [ETA: {left_time}]")

    print(f"Sampling complete [AVG Time: {np.mean(elapsed_time_list):.2f}s]")
    print(f"All results have been saved to {save_dir}.")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
    )
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    parser.add_argument("--device", default=0, type=int, help='the cuda device to use to generate images')
    parser.add_argument("--model_path", default="", type=str, help='path of pretrained diffusion model')
    parser.add_argument("--data_pth", default="", type=str, help='path of input noisy image')
    parser.add_argument("--gt_pth", default=None, help='path of reference HSNR image')
    parser.add_argument("--save_dir", default="./results", type=str, help='dir to save generated results')

    parser.add_argument("--ae_iters", default=10, type=int, help='iters for updating autoencoder per timestep')
    parser.add_argument("--ae_lr", default=1e-1, type=float, help='learning rate for updating autoencoder per timestep')
    parser.add_argument("--sigma", default=1, type=float, help='hyperparameter sigma for updating autoencoder per timestep')
    parser.add_argument("--omega", default=10, type=float, help='factor used to scale the guidance strength')

    parser.add_argument("--max_size", default=2048, type=int)
    parser.add_argument("--cropped_size", default=0, type=int)
    parser.add_argument("--num_sample", default=1, type=int)
    parser.add_argument("--use_ddim", action='store_true')

    return parser


if __name__ == "__main__":
    main()
