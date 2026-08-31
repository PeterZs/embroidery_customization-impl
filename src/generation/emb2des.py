import json
import os
from PIL import Image
import torch
import cv2
import numpy as np
from controlnet_aux import HEDdetector
from diffusers.utils import load_image
import argparse

from tqdm import tqdm

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
os.environ['PYTHONPATH'] = str(project_root)
sys.path.append(str(project_root))

from src.common.load_pipeline import load_sd3_pipeline
from src.common.wd14tagger import WD14Tagger
from src.common.controlnet_detector import tile_detector


def parse_args():
    parser = argparse.ArgumentParser(description="Process image generation settings and paths.")

    # Prompt and Negative Prompt
    parser.add_argument(
        '--prompt', 
        type=str, 
        default=(
            "Flat design, vector design, digital design, cartoon design,"
            "High quality, masterpiece, Flat design style, clean lines, "
            "uniform color blocks, smooth surface, vector graphic effect"
        ),
        help="Prompt for the image generation"
    )
    
    parser.add_argument(
        '--negative_prompt', 
        type=str,
        default=(
            'embroidery, lowres, bad anatomy, bad hands, missing fingers, '
            'cropped, worst quality, low quality'
        ),
        help="Negative prompt to avoid unwanted features"
    )

    # Inference Settings
    parser.add_argument('--seed', type=int, default=42, help="Seed")
    parser.add_argument('--num_inference_steps', type=int, default=28, help="Number of inference steps")
    parser.add_argument('--cfg', type=float, default=4.0, help="Classifier-free guidance scale")

    # ControlNet Parameters
    parser.add_argument('--controlnet_canny_scale', type=float, default=0.3, help="Scale for ControlNet Canny")
    parser.add_argument('--controlnet_tile_scale', type=float, default=0.8, help="Scale for ControlNet Tile")
    parser.add_argument('--blur_strength', type=int, default=7, help="Blur strength for Gaussian blur")

    # Paths
    parser.add_argument(
        '--input_folder', 
        type=str, 
        default="",
        help="Folder containing input images"
    )
    parser.add_argument(
        '--output_folder', 
        type=str, 
        default="",
        help="Folder to save output images"
    )

    parser.add_argument(
        '--device', 
        type=str, 
        default="cuda",
        help="Device"
    )

    parser.add_argument(
        '--wd14_models_dir', 
        type=str, 
        required=True,
    )

    parser.add_argument(
        '--sd3_path', 
        type=str, 
        default="stabilityai/stable-diffusion-3-medium-diffusers",
    )
    parser.add_argument(
        '--controlnet_canny_path', 
        type=str, 
        default="InstantX/SD3-Controlnet-Canny" ,
    )
    parser.add_argument(
        '--controlnet_tile_path', 
        type=str, 
        default="InstantX/SD3-Controlnet-Tile",
    )
    parser.add_argument(
        '--hed_model_dir', 
        type=str, 
        default="lllyasviel/Annotators",
    )

    return parser.parse_args()


def preprocess_image(image, args):
    image = np.array(image)

    # preprocess for controlnet-canny
    hed = HEDdetector.from_pretrained(args.hed_model_dir)
    canny_img = hed(image, resolution=1024, safe=True)
    canny_img = canny_img.resize((512, 512), Image.Resampling.LANCZOS)
    canny_img = canny_img.resize((1024, 1024), Image.Resampling.LANCZOS)

    tile_map = tile_detector(image, mode='blur', blur_strength=args.blur_strength)

    return [canny_img, tile_map]


def generate(pipe, image, args):
    resized_image = image.resize(size=(1024, 1024))

    # wd14 tagger
    tagger = WD14Tagger(args.wd14_models_dir)   
    tags = tagger.tag_image(image, model_name="wd-convnext-tagger-v3", threshold=0.35)
    prompt = args.prompt + "," + tags
    print(prompt)
    
    # preprocess for controlnet
    controlnet_imgs = preprocess_image(resized_image, args)

    generator = torch.Generator(pipe.device).manual_seed(args.seed)
    generated_image = pipe(
        prompt=prompt,
        prompt_2=prompt,
        negative_prompt=args.negative_prompt,
        control_image=controlnet_imgs,
        generator=generator,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.cfg,
        controlnet_conditioning_scale=[float(args.controlnet_canny_scale), float(args.controlnet_tile_scale)],
    ).images[0]

    return generated_image, prompt, resized_image, controlnet_imgs


if __name__ == '__main__':
    args = parse_args()
    pipe = load_sd3_pipeline(
        sd3_path=args.sd3_path,
        controlnet_canny_path=args.controlnet_canny_path,
        controlnet_tile_path=args.controlnet_tile_path,
        device=args.device
        )

    os.makedirs(args.output_folder, exist_ok=True)

    image_filenames = [f for f in os.listdir(args.input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_filenames)
    print(f"Total images: {total_images}")

    for idx, image_filename in enumerate(tqdm(image_filenames, desc="Processing Images", unit="image"), 1):
        image_path = os.path.join(args.input_folder, image_filename)
        input_image = load_image(image_path)

        generated_image, prompt, resized_image, controlnet_imgs = generate(pipe, input_image, args)

        generated_image.save(os.path.join(args.output_folder, f"{image_filename.split('.')[0]}.png"))