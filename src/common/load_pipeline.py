import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    SD3ControlNetModel,
    SD3MultiControlNetModel,
    StableDiffusion3ControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
)

from src.prompt_routing.custom_modules import replace_attn_processor


def get_torch_device(device):
    return torch.device(device if torch.cuda.is_available() else "cpu")


def load_sdxl_scheduler(sdxl_path):
    return EulerAncestralDiscreteScheduler.from_pretrained(sdxl_path, subfolder="scheduler")


def load_sdxl_vae(vae_path):
    return AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)


def load_sdxl_canny_tile_i2i_pipeline(
    sdxl_path,
    vae_path,
    controlnet_canny_path,
    controlnet_tile_path,
    device="cuda",
):
    device = get_torch_device(device)
    eulera_scheduler = load_sdxl_scheduler(sdxl_path)
    controlnet_canny = ControlNetModel.from_pretrained(controlnet_canny_path, torch_dtype=torch.float16)
    controlnet_tile = ControlNetModel.from_pretrained(controlnet_tile_path, torch_dtype=torch.float16)
    vae = load_sdxl_vae(vae_path)

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        sdxl_path,
        controlnet=[controlnet_canny, controlnet_tile],
        vae=vae,
        torch_dtype=torch.float16,
        scheduler=eulera_scheduler
    )
    pipe.to(device)
    return pipe


def load_sdxl_i2i_pipeline(
    sdxl_path,
    vae_path,
    device="cuda",
    use_prompt_separation=False,
    style_blocks="4block"
):  
    device = get_torch_device(device)
    eulera_scheduler = load_sdxl_scheduler(sdxl_path)
    vae = load_sdxl_vae(vae_path)

    if use_prompt_separation:
        from src.prompt_routing.custom_img2img_pipe import StableDiffusionXLImg2ImgPipeline    
    else:   
        from diffusers import StableDiffusionXLImg2ImgPipeline
    
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        sdxl_path,
        vae=vae,
        torch_dtype=torch.float16,
        scheduler=eulera_scheduler
    )
    if use_prompt_separation:
        pipe.unet = replace_attn_processor(pipe.unet, style_blocks=style_blocks)

    pipe.to(device)
    return pipe


def load_sdxl_t2i_pipeline(
    sdxl_path,
    vae_path,
    device="cuda",
    use_prompt_separation=False,
    style_blocks="4block"
):
    device = get_torch_device(device)
    eulera_scheduler = load_sdxl_scheduler(sdxl_path)
    vae = load_sdxl_vae(vae_path)

    if use_prompt_separation:
        from src.prompt_routing.custom_t2i_pipe import StableDiffusionXLPipeline
    else:
        from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        sdxl_path,
        vae=vae,
        torch_dtype=torch.float16,
        scheduler=eulera_scheduler
    )
    if use_prompt_separation:
        pipe.unet = replace_attn_processor(pipe.unet, style_blocks=style_blocks)
    
    pipe.to(device)
    return pipe


def load_sd3_pipeline(
    sd3_path,
    controlnet_canny_path,
    controlnet_tile_path,
    device="cuda"
):
    device = get_torch_device(device)

    controlnet_canny = SD3ControlNetModel.from_pretrained(controlnet_canny_path, torch_dtype=torch.float16)
    controlnet_tile = SD3ControlNetModel.from_pretrained(controlnet_tile_path, torch_dtype=torch.float16)

    controlnet = SD3MultiControlNetModel([controlnet_canny, controlnet_tile])

    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        sd3_path,
        text_encoder_3=None,  # no-T5
        tokenizer_3=None,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    pipe.to(device)
    return pipe
