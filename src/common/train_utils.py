import os
import subprocess

STYLE_BLOCK_DICT = {
    # One randomly selected block
    "up_0_1": ["up_blocks.0.attentions.1"],
    "up_0_2": ["up_blocks.0.attentions.2"],
    "down_1_1": ["down_blocks.1.attentions.1"],
    "down_2_0": ["down_blocks.1.attentions.1"],

    # Two blocks with the lowest similarity
    "2blocks": [
        "down_blocks.1.attentions.1", 
        "up_blocks.0.attentions.2"
    ],

    "2blocks_1": [
        "down_blocks.1.attentions.1", 
        "down_blocks.2.attentions.0",
    ],
    "2blocks_2": [
        "up_blocks.0.attentions.1",
        "up_blocks.0.attentions.2"
    ],
    "2blocks_3": [
        "down_blocks.2.attentions.1",
        "mid_block.attentions.0"
    ],

    # Randomly selected four-block combinations
    "4blocks_0": [
        "down_blocks.1.attentions.0",
        "down_blocks.1.attentions.1",
        "down_blocks.2.attentions.0",
        "down_blocks.2.attentions.1"
    ],
    "4blocks_1": [
        "up_blocks.0.attentions.1",
        "up_blocks.0.attentions.2",
        "up_blocks.1.attentions.0",
        "up_blocks.1.attentions.1"
    ],
    "4blocks_2": [
        "down_blocks.1.attentions.0",
        "down_blocks.2.attentions.1",    
        "up_blocks.0.attentions.0",
        "up_blocks.1.attentions.0"
    ],

    # Currently selected four blocks
    "4blocks": [
        "down_blocks.1.attentions.1", 
        "down_blocks.2.attentions.0",
        "up_blocks.0.attentions.1",
        "up_blocks.0.attentions.2"
    ],

    # Six blocks with the lowest similarity
    "6blocks_0": [
        "down_blocks.1.attentions.1", 
        "down_blocks.2.attentions.0",
        "down_blocks.2.attentions.1",   
        "mid_block.attentions.0",
        "up_blocks.0.attentions.1",
        "up_blocks.0.attentions.2"
    ],

    # Six randomly selected blocks
    "6blocks_1": [
        "down_blocks.1.attentions.0",
        "down_blocks.1.attentions.1",
        "down_blocks.2.attentions.1",
        "up_blocks.0.attentions.2",
        "up_blocks.1.attentions.1",
        "mid_block.attentions.0"
    ],

    "6blocks_2": [
        "down_blocks.1.attentions.0",
        "down_blocks.1.attentions.1",
        "up_blocks.0.attentions.0",
        "up_blocks.0.attentions.1",
        "up_blocks.0.attentions.2",
        "up_blocks.1.attentions.0",
    ],


    # All blocks
    "all": [
        "down_blocks.1.attentions.0",
        "down_blocks.1.attentions.1",
        "down_blocks.2.attentions.0",
        "down_blocks.2.attentions.1",
        "up_blocks.0.attentions.0",
        "up_blocks.0.attentions.1",
        "up_blocks.0.attentions.2",
        "up_blocks.1.attentions.0",
        "up_blocks.1.attentions.1",
        "up_blocks.1.attentions.2",
        "mid_block.attentions.0"
    ]
}

# Taken (and slightly modified) from B-LoRA repo https://github.com/yardenfren1996/B-LoRA/blob/main/blora_utils.py
def is_belong_to_blocks(key, blocks):
    try:
        for g in blocks:
            if g in key:
                return True
        return False
    except Exception as e:
        raise type(e)(f"failed to is_belong_to_block, due to: {e}")


def get_unet_lora_target_modules(unet, use_blora, target_blocks=None):
    if use_blora:
        content_b_lora_blocks = "unet.up_blocks.0.attentions.0"
        style_b_lora_blocks = "unet.up_blocks.0.attentions.1"
        target_blocks = [content_b_lora_blocks, style_b_lora_blocks]
    try:
        blocks = [(".").join(blk.split(".")[1:]) for blk in target_blocks]

        attns = [
            attn_processor_name.rsplit(".", 1)[0]
            for attn_processor_name, _ in unet.attn_processors.items()
            if is_belong_to_blocks(attn_processor_name, blocks)
        ]

        return [f"{attn}.{mat}" for mat in ["to_k", "to_q", "to_v", "to_out.0"] for attn in attns]
    except Exception as e:
        raise type(e)(
            f"failed to get_target_modules, due to: {e}. "
            f"Please check the modules specified in --lora_unet_blocks are correct"
        )


def _resolve_style_blocks(style_blocks):
    if style_blocks is None:
        return ()
    if isinstance(style_blocks, str):
        try:
            return tuple(STYLE_BLOCK_DICT[style_blocks])
        except KeyError as exc:
            valid_keys = ", ".join(sorted(STYLE_BLOCK_DICT))
            raise ValueError(f"Unknown style_blocks={style_blocks!r}. Valid keys: {valid_keys}") from exc
    return tuple(style_blocks)


def set_lora_trainable(unet, train_scope, style_blocks=None):
    """Set trainable content-style LoRA parameters.

    train_scope:
        "style": train selected style blocks only
        "content": train non-style blocks only
        "all": train all content-style LoRA parameters
        "none": freeze all content-style LoRA parameters
    """
    valid_scopes = {"style", "content", "all", "none"}
    if train_scope not in valid_scopes:
        raise ValueError(f"train_scope must be one of {sorted(valid_scopes)}, got {train_scope!r}")

    resolved_style_blocks = _resolve_style_blocks(style_blocks)

    stats = {"trainable": 0, "frozen": 0}

    for name, parameter in unet.named_parameters():
        if "content_style_lora" not in name:
            continue

        is_style_block = any(block in name for block in resolved_style_blocks)
        should_train = (
            train_scope == "all"
            or (train_scope == "style" and is_style_block)
            or (train_scope == "content" and not is_style_block)
        )
        parameter.requires_grad_(should_train)
        stats["trainable" if should_train else "frozen"] += parameter.numel()

    return stats


def create_and_run_sh_script(sh_file_path, sh_file_content):
    """
    Create a .sh file, set permissions, run the script, and stream output to the terminal.

    Args:
        sh_file_path (str): Full path to the .sh file.
        sh_file_content (str): Content to write to the .sh file.

    Returns:
        int: Return code from the run. 0 means success; non-zero means failure.
    """
    with open(sh_file_path, 'w') as sh_file:
        sh_file.write(sh_file_content)

    os.chmod(sh_file_path, 0o755)

    try:
        result = subprocess.run([sh_file_path], stdout=None, stderr=None)
        return result.returncode
    except Exception as e:
        return f"Error while executing script: {e}"
