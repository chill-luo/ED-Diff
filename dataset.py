import os
import numpy as np
import blobfile as bf
import tifffile as tiff

from torch.utils.data import Dataset


class TifDataset(Dataset):
    def __init__(self, data_dir, gt_dir=None, max_size=10000):
        super().__init__()
        if "," in data_dir:
            self.local_images = []

            if gt_dir is None:
                gt_dir = [None] * len(data_dir.split(","))
            else:
                assert len(gt_dir.split(",")) == len(data_dir.split(",")), "The number of gt_dir should be the same as data_dir."
            
            for d, g in zip(data_dir.split(","), gt_dir.split(",")):
                self.local_images.extend(self._list_image_files(d, g, max_size))
        else:
            self.local_images = self._list_image_files(data_dir, gt_dir, max_size)

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        patch, patch_gt, pmin, pmax, name = self.local_images[idx]
        arr = patch.astype(np.float32)
        
        if pmin != pmax:
            arr = 2 * (arr - pmin) / (pmax - pmin) - 1

        return arr[None, ...], patch_gt[None, ...].astype(np.float32), pmin, pmax, name
    
    def _list_image_files(self, data_dir, gt_dir=None, max_size=10000):
        results = []
        print(f"Scanning {data_dir}...")
        for entry in sorted(bf.listdir(data_dir)):
            full_path = bf.join(data_dir, entry)
            if bf.isdir(full_path) or not entry.endswith(".tif"):
                continue

            stack = tiff.imread(full_path)
            if stack.ndim == 2:
                stack = stack[None, ...]
            if gt_dir is not None:
                gt_path = bf.join(gt_dir, entry)
                gt = tiff.imread(gt_path)
                if gt.ndim == 2:
                    gt = gt[None, ...]
            else:
                gt = np.zeros_like(stack)
            
            n, h, w = stack.shape
            for pn in range(n):
                h_new = min(h - h % 32, max_size)
                w_new = min(w - w % 32, max_size)
                h_start = (h - h_new) // 2
                w_start = (w - w_new) // 2

                p1 = int(stack[pn, h_start:h_start+h_new, w_start:w_start+w_new].min())
                p2 = int(stack[pn, h_start:h_start+h_new, w_start:w_start+w_new].max())
                results.append((
                    stack[pn, h_start:h_start+h_new, w_start:w_start+w_new], 
                    gt[pn, h_start:h_start+h_new, w_start:w_start+w_new],
                    p1, 
                    p2,
                    entry.replace(".tif", "")))
            
        print(f"Scanning done, there are {len(results)} samples for sampling.")
        return results
