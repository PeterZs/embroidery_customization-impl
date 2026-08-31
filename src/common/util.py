import json
import os
import numpy as np
from PIL import Image

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def save_config(args, config_path, **extra_params):
    config = vars(args)
    config.update(extra_params)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")


def print_config(args):
    config = vars(args)
    print("Current Configuration:")
    print(json.dumps(config, indent=4))


def get_image_paths(folder_path):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

    image_paths = [
        os.path.join(folder_path, file) for file in os.listdir(folder_path)
        if os.path.splitext(file)[1].lower() in valid_extensions
    ]

    return image_paths


def color_fix(img1, img2):
    """Combine the brightness channel of img1 with the color channels of img2"""
    img2 = img2.resize(img1.size, Image.LANCZOS)
    
    lab1 = img1.convert('LAB')
    lab2 = img2.convert('LAB')

    L1, _, _ = lab1.split()
    _, A2, B2 = lab2.split()

    combined_lab = Image.merge('LAB', (L1, A2, B2))

    combined_rgb = combined_lab.convert('RGB')

    return combined_rgb


def resize_img(img_pil):
    """Image Resize by Longer Side"""
    size = 1024
    w, h = img_pil.size
    if h > w:
        width = size * w // h
        height = size
    else:
        width = size
        height = size * h // w

    new_width = (width//8)*8
    new_height = (height//8)*8

    resized_img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
    return resized_img_pil
