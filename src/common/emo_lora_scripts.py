import shlex


def shell_quote(value):
    return shlex.quote(str(value))


def build_remove_emb_script(gpu_id, model_config, input_folder, output_folder):
    q = shell_quote
    return f"""#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES={q(gpu_id)}
python -m src.generation.emb2des \\
    --input_folder={q(input_folder)} \\
    --output_folder={q(output_folder)} \\
    --sd3_path={q(model_config["sd3_path"])} \\
    --controlnet_canny_path={q(model_config["sd3_controlnet_canny_path"])} \\
    --controlnet_tile_path={q(model_config["sd3_controlnet_tile_path"])} \\
    --hed_model_dir={q(model_config["hed_model_dir"])} \\
    --controlnet_canny_scale=0.2 \\
    --controlnet_tile_scale=0.8 \\
    --blur_strength=15 \\
    --wd14_models_dir={q(model_config['wd14_models_dir'])}
    """


def build_stage1_train_script(
    gpu_id,
    model_config,
    stage1_steps,
    stage1_checkpoint_dir,
    base_content_dir,
    base_style_dir,
    style_blocks,
):
    q = shell_quote
    return f"""#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES={q(gpu_id)}
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
accelerate launch --gpu_ids={q(gpu_id)} \\
src/training/train_emo_lora_stage1.py \\
    --pretrained_model_name_or_path={q(model_config['sdxl_path'])} \\
    --pretrained_vae_model_name_or_path={q(model_config['vae_path'])} \\
    --resolution=1024 \\
    --train_batch_size=1 \\
    --max_train_steps={stage1_steps} \\
    --checkpointing_steps=100\\
    --learning_rate=1e-4 --lr_scheduler="constant" --lr_warmup_steps=0 \\
    --mixed_precision="fp16" \\
    --output_dir={q(stage1_checkpoint_dir)} \\
    --seed=42 \\
    --rank=32 \\
    --instance_data_root_content={q(base_content_dir)} \\
    --instance_prompt_content={q("a <des>")} \\
    --instance_data_root_style={q(base_style_dir)} \\
    --instance_prompt_style={q("a <des> in [emb] style")} \\
    --resume_from_checkpoint="latest"  \\
    --style_blocks={q(style_blocks)} \\
    """


def build_expand_style_images_script(
    gpu_id,
    model_config,
    lora_path,
    output_folder,
    prompts_json,
):
    q = shell_quote
    return f"""#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES={q(gpu_id)}
python -m src.generation.text2img \\
    --sdxl_path={q(model_config['sdxl_path'])} \\
    --vae_path={q(model_config['vae_path'])} \\
    --lora_path={q(lora_path)} \\
    --lora_scale=1.0 \\
    --output_folder={q(output_folder)} \\
    --num_images_per_prompt=1 \\
    --prompts_json={q(prompts_json)} \\
    --use_text2patch \\
        """


def build_stage2_train_script(
    gpu_id,
    model_config,
    stage2_steps,
    stage2_checkpoint_dir,
    base_dataset_dir,
    expand_dataset_dir,
    loss_contrastive_weight,
    loss_style_pred_expand_weight,
    style_blocks,
):
    q = shell_quote
    return f"""#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES={q(gpu_id)}
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
accelerate launch --gpu_ids={q(gpu_id)} \\
src/training/train_emo_lora_stage2.py \\
    --pretrained_model_name_or_path={q(model_config['sdxl_path'])} \\
    --pretrained_vae_model_name_or_path={q(model_config['vae_path'])} \\
    --resolution=1024 \\
    --train_batch_size=1 \\
    --max_train_steps={stage2_steps} \\
    --checkpointing_steps=100 \\
    --learning_rate=1e-4 --lr_scheduler="constant" --lr_warmup_steps=0 \\
    --mixed_precision="fp16" \\
    --output_dir={q(stage2_checkpoint_dir)} \\
    --seed=42 \\
    --rank=32 \\
    --resume_from_checkpoint="latest" \\
    --train_data_dir_base={q(base_dataset_dir)} \\
    --train_data_dir_expand={q(expand_dataset_dir)} \\
    --content_prompt_base={q("a <des>")} \\
    --contrastive_mode="all" \\
    --loss_contrastive_weight={loss_contrastive_weight} \\
    --loss_style_pred_base_weight=1.0  --loss_style_pred_expand_weight={loss_style_pred_expand_weight} \\
    --style_blocks={q(style_blocks)}
        """
