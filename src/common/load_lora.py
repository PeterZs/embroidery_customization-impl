from diffusers.loaders import StableDiffusionLoraLoaderMixin

def load_emolora(pipe, lora_path):
    """
    Load a LoRA model into specific UNet blocks of the pipeline.
    """
    target_blocks = [
        "unet.down_blocks.1.attentions.1",
        "unet.down_blocks.2.attentions.0",
        "unet.up_blocks.0.attentions.1",
        "unet.up_blocks.0.attentions.2"
    ]

    pipe.unload_lora_weights()
    state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(lora_path)
    filtered_state_dict = {k: v for k, v in state_dict.items() if any(block in k for block in target_blocks)}
    StableDiffusionLoraLoaderMixin.load_lora_into_unet(filtered_state_dict, None, pipe.unet)

    return pipe
