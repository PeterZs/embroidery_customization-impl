import os
import torch
import argparse
from tqdm import tqdm
from src.common.load_pipeline import load_sdxl_t2i_pipeline, load_sdxl_i2i_pipeline
import json
from rembg import remove

def parse_args():
    parser = argparse.ArgumentParser(description="Process image generation settings and paths.")

    # Model paths
    parser.add_argument('--sdxl_path', type=str, required=True,
                        help="Path to the SDXL model")
    parser.add_argument('--vae_path', type=str, required=True,
                        help="Path to the VAE model")
    
    parser.add_argument('--lora_path', type=str, default=None,
                        help="Path to the LoRA model, if available")
       
    # Output folder path
    parser.add_argument('--output_folder', type=str, required=True,
                        help="Folder containing output images")

    # Inference parameters
    parser.add_argument('--device', type=str, default="cuda",
                        help="Device to run the models on")
    parser.add_argument('--negative_prompt', type=str,
                        default='lowres, bad anatomy, cropped, worst quality, low quality',
                        help="Negative prompt to avoid unwanted features")
    parser.add_argument('--seed', type=int, default=42, help="Seed for randomness")
    parser.add_argument('--num_inference_steps', type=int, default=30, help="Number of inference steps")
    parser.add_argument('--cfg', type=float, default=7.0, help="Classifier-free guidance scale")
    parser.add_argument('--lora_scale', type=float, default=1.0, help="Scale for Lora")

    parser.add_argument('--num_images_per_prompt', type=int, default=None,
                        help="Number of images to generate per prompt")
    
    parser.add_argument("--content_prompt", type=str, default=None,)
    parser.add_argument('--prompts_json', type=str, default=None,
                        help="Path to json file containing prompts")

    parser.add_argument("--use_text2patch", action="store_true")

    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    # load pipe
    pipe = load_sdxl_t2i_pipeline(
        args.sdxl_path,
        args.vae_path,
        device=args.device,
        use_prompt_separation=False
        )
    
    pipe.enable_vae_slicing()

    # prompts
    if args.content_prompt is not None:
        prompts = [args.content_prompt]
    elif args.prompts_json is not None:
        with open(args.prompts_json, "r", encoding="utf-8") as file:
            prompts = json.load(file)
    else:
        prompts = [
            "A dog",
            "A robot",
            "A car",
        ]

    generator = torch.Generator(pipe.device).manual_seed(args.seed)

    # text2patch
    if args.use_text2patch:
        init_images = []
        for prompt in prompts:
            formatted_prompt = f"""
an embroidery patch of a cartoon {prompt}, clear lines,uniform color, simple design, 
white background, solo, best quality """

            with torch.no_grad():
                init_image = pipe(
                    prompt=formatted_prompt,
                    num_inference_steps=30,
                    guidance_scale=5,
                    generator=generator,
                ).images[0]
                init_image = remove(init_image, bgcolor=(255, 255, 255, 255)).convert("RGB")
            
            init_images.append(init_image)

        del pipe
        torch.cuda.empty_cache()

        pipe = load_sdxl_i2i_pipeline(
            args.sdxl_path,
            args.vae_path,
            device=args.device,
            use_prompt_separation=False
        )
    else:
        init_images = [None] * len(prompts)


    # load lora
    if args.lora_path:
        pipe.unload_lora_weights()
        pipe.load_lora_weights(args.lora_path)

    # Set the output folder path.
    os.makedirs(args.output_folder, exist_ok=True)

    with torch.no_grad():
        for prompt, init_image in tqdm(zip(prompts, init_images), desc="Generating images", unit="prompt"):
            style_prompt = f"{prompt} in [emb] style"

            if args.use_text2patch:
                generated_images = pipe(
                    prompt=style_prompt,
                    negative_prompt=args.negative_prompt,
                    generator=generator,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.cfg,
                    cross_attention_kwargs={"scale": args.lora_scale},
                    num_images_per_prompt=args.num_images_per_prompt,
                    image=init_image,
                    strength=1.0,
                ).images
            else:
                generated_images = pipe(
                    prompt=style_prompt,
                    negative_prompt=args.negative_prompt,
                    generator=generator,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.cfg,
                    cross_attention_kwargs={"scale": args.lora_scale},
                    num_images_per_prompt=args.num_images_per_prompt,
                ).images

            for i, generated_image in enumerate(generated_images):
                image_filename = f"{prompt.lower().replace(' ', '_')}_{i}.png"
                generated_image.save(os.path.join(args.output_folder, image_filename))

    del pipe
    torch.cuda.empty_cache()
