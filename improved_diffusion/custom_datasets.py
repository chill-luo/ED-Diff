import os
import ipdb
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


def load_Tiffdata(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir, image_size)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [0] * len(all_files)
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = CustomDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir, image_size):
    results = []
    print("Scanning dataset dir...")
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        if bf.isdir(full_path):
            continue
        ext = entry.split(".")[-1]
        
        stack = tiff.imread(full_path)
        n, h, w = stack.shape
        for pn in range(n):
            p1, p2 = int(stack[pn].min()), int(stack[pn].max())
            for ph in range(0, h, image_size):
                for pw in range(0, w, image_size):
                    if pw + image_size > w:
                        pw = w - image_size
                    if ph + image_size > h:
                        ph = h - image_size
                    results.append((stack[pn, ph:ph+image_size, pw:pw+image_size], p1, p2))
        
    print(f"Scanning done, there are {len(results)} samples for training.")
    return results


class CustomDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        patch, p1, p2 = self.local_images[idx]

        arr = patch.astype(np.float32)
        if p1 is not None:
            arr = 2 * (arr - p1) / (p2 - p1) - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return arr[None, ...], out_dict
