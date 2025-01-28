import torch
import os
import numpy as np
import nibabel as nib
import albumentations as A
import cv2
from torch.utils.data import Dataset, DataLoader
import random
import time
from tqdm import tqdm
import h5py

def varifyh5(filename): # read the h5 file to see if the conversion is finished or not
    try:
        with h5py.File(filename, "r") as hf:   # can read successfully
            pass
        return True
    except OSError:     # transform not complete
        return False


def load_CT_slice(ct_path, slice_idx=None):
    """For AbdomenAtlasPro data: ranging from [-1000, 1000], shape of (H W D) """
    with h5py.File(ct_path, 'r') as hf:
        nii = hf['image']
        z_shape = nii.shape[2]

        # NOTE: take adjacent 3 slices into the 3 RGB channel
        if slice_idx is None:
            slice_idx = random.randint(0, z_shape - 3)   # `random.randint` includes end point
        while True:
            try:    # some slices of some CT are broken
                ct_slice = nii[:, :, slice_idx:slice_idx + 3]   
                break
            except: # if broken, randomly select until select the non-broken slice
                print(f"\033[31mBroken slice: {ct_path.split('/')[-2]}, slice {slice_idx}\033[0m")
                slice_idx = random.randint(0, z_shape - 3)

    # ct_slice = np.repeat(ct_slice, repeats=3, axis=2)    # (H W 1) -> (H W 3)
            
    # target range: [-1000, 1000] -> [-1, 1]
    ct_slice[ct_slice > 1000.] = 1000.    # clipping range and normalize
    ct_slice[ct_slice < -1000.] = -1000.
    ct_slice = (ct_slice + 1000.) / 2000.       # [-1000, 1000] --> [0, 1]
    return ct_slice  # (H W 3)[0, 1]

class HWCarrayToCHWtensor(A.ImageOnlyTransform):
    """Converts (H, W, C) NumPy array to (C, H, W) PyTorch tensor."""
    def apply(self, img, **kwargs):
        return torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) → (C, H, W)

class CTDataset(Dataset):
    def __init__(self, file_paths, image_transform=None, tokenizer=None):
        """ (training only)
        Args:
            file_paths (list): List of paths to 3D CT volumes (.nii.gz).
            transform (albumentations.Compose): Transformations to apply to 2D slices.
        """
        self.file_paths = file_paths
        self.image_transform = image_transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        ct_path = self.file_paths[idx]  # random CT
        ct_slice = load_CT_slice(os.path.join(ct_path, "ct.h5"))    # random slice
        ct_slice = self.image_transform(image=ct_slice)["image"]  # data augmentation
        if ct_path.split("/")[-2].startswith("BDMAP_A"):
            text_prompt = "An Arterial CT slice."
        elif ct_path.split("/")[-2].startswith("BDMAP_V"):
            text_prompt = "A Portal-venous CT slice."
        else:
            raise NotImplementedError("Only support Arterial (BDMAP_A) and Portal-venous (BDMAP_V) right now.")

        example = dict()
        example["pixel_values"] = ct_slice
        example["input_ids"] = self.tokenize_caption(text_prompt)

        return example  # Shape: (C, H, W)
    
    def tokenize_caption(self, text_prompt, is_train=True):
        captions = text_prompt
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids 

if __name__ == "__main__":
    train_data_dir = "/mnt/T9/AbdomenAtlas/image_mask_h5"
    paths = sorted([entry.path for entry in os.scandir(train_data_dir)
                            if entry.name.startswith("BDMAP_A") or entry.name.startswith("BDMAP_V")])
    paths = [entry.path.replace("ct.h5", "") for path in  paths
                                            for entry in os.scandir(path) if entry.name == "ct.h5"]
    print(len(paths), "CT scans found!")


    train_transforms = A.Compose([
        A.Resize(512, 512, interpolation=cv2.INTER_LINEAR),
        A.RandomResizedCrop((512, 512), scale=(0.75, 1.0), ratio=(1., 1.), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        HWCarrayToCHWtensor(p=1.),
    ])

    train_dataset = CTDataset(paths, transform=train_transforms)

    def collate_fn(examples):
        # pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = torch.stack([example for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        # input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values}#, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=1,
        pin_memory=True
    )

    for batch in tqdm(train_dataloader):
        batch = batch["pixel_values"]