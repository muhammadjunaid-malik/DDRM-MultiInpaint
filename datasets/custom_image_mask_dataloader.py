from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np


def center_crop_arr(pil_image, image_size=256, is_mask=False):
    # Resample method
    downsample_resample = Image.NEAREST if is_mask else Image.BOX
    upsample_resample = Image.NEAREST if is_mask else Image.BICUBIC

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=downsample_resample
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=upsample_resample
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


class CenterCropTransform:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image, mask):
        image_arr = center_crop_arr(image, self.image_size, is_mask=False)
        mask_arr = center_crop_arr(mask, self.image_size, is_mask=True)

        # Convert to tensors
        image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).float() / 255.0  # if binary mask

        return image_tensor, mask_tensor



class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_size=255):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        image_names = sorted(os.listdir(image_dir))
        # Keep only image names that exist in both directories
        self.image_names = [
            name for name in image_names
            if os.path.exists(os.path.join(mask_dir, name))
        ]
        print(f"Found {len(self.image_names)} image-mask pairs.")
        if transform :
            self.transform = transform
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        try:
            image = Image.open(image_path)
            mask = Image.open(mask_path)
        except Exception as e:
            print(f"Error loading image or mask for {image_name}: {e}")
            raise e

        if self.transform:
            image,mask = self.transform(image,mask)
        
        return image, mask 