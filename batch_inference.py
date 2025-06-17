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

    print("Creating dataset and models...")
    dataset = TifDataset(args.data_dir, gt_dir=args.gt_dir, max_size=args.max_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

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
    save_args_to_yaml(args, os.path.join(save_dir, "args.yaml"))
    print(f"Target directory: {save_dir}.")

    elapsed_time_list = []
    print("Start sampling...")
    for ind, data in enumerate(dataloader):
        noisy, gt, p1, p2, name = data
        noisy = noisy.repeat(args.num_sample, 1, 1, 1).cuda()      # B 1 H W
        p1, p2 = int(p1), int(p2)

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
        tiff.imwrite(os.path.join(save_dir, "output", f"{str(ind+1).zfill(5)}_{name[0]}.tif"), denoised[0, 0, :, :].astype(np.uint16))

        def u16tou8(x):
            x = x.astype(np.float32)
            x = (x - x.min()) / (x.max() - x.min()) * 255
            return x.clip(0, 255).astype(np.uint8)

        denoised_vis = u16tou8(denoised[0, 0, :, :])
        noisy_vis = u16tou8(noisy[0, 0, :, :])
        if args.gt_dir is not None:
            gt = gt.cpu().numpy().clip(0, 65525).astype(np.uint16)
            gt_vis = u16tou8(gt[0, 0, :, :])
            tiff.imwrite(os.path.join(save_dir, f"vis_{str(ind+1).zfill(5)}_{name[0]}.tif"), np.hstack((noisy_vis, denoised_vis, gt_vis)).astype(np.uint8))
        else:
            tiff.imwrite(os.path.join(save_dir, f"vis_{str(ind+1).zfill(5)}_{name[0]}.tif"), np.hstack((noisy_vis, denoised_vis)).astype(np.uint8))

        left_time = datetime.timedelta(seconds=int(np.mean(elapsed_time_list) * (len(dataloader) - ind - 1)))
        print(f"Created {str(ind+1).zfill(5)}_{name[0]} within {elapsed_time:.2f}s. [ETA: {left_time}]")

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
    parser.add_argument("--data_dir", default="", type=str, help='dir of input noisy images')
    parser.add_argument("--gt_dir", default=None, help='dir of reference HSNR images')
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
