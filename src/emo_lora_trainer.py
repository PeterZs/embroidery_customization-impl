import os
import argparse
import random
import json
import sys
from pathlib import Path

from PIL import Image

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
os.environ["PYTHONPATH"] = str(project_root)

from src.common.train_utils import create_and_run_sh_script
from src.common.io_utils import clear_folder, copy_folder_contents, image_files, subject_from_filename
from src.common.emo_lora_scripts import (
    build_expand_style_images_script,
    build_remove_emb_script,
    build_stage1_train_script,
    build_stage2_train_script,
)


class EmoLoRATrainer:
    STAGE1_STEPS = 400
    STAGE2_MAX_TRAIN_STEPS = 600
    LOSS_CONTRASTIVE_WEIGHT = 0.0001
    LOSS_STYLE_PRED_EXPAND_WEIGHT = 0.1
    STYLE_BLOCKS = "4blocks"
    IMAGE_SIZE = (1024, 1024)
    REQUIRED_MODEL_CONFIG_KEYS = {
        "sdxl_path",
        "vae_path",
        "sd3_path",
        "sd3_controlnet_canny_path",
        "sd3_controlnet_tile_path",
        "hed_model_dir",
        "wd14_models_dir",
    }
    EXPAND_PROMPTS = [
        "A yellow dog",
        "A silver robot",
        "A red car",
    ]

    def __init__(
        self,
        model_config,
        basename,
        train_image_path,
        output_dir,
        gpu_id=0,
    ):
        self.model_config = model_config
        self.validate_model_config(model_config)
        self.gpu_id = str(gpu_id)
        self.stage1_steps = self.STAGE1_STEPS
        self.stage2_steps = self.STAGE2_MAX_TRAIN_STEPS
        
        self.output_dir = output_dir
        self.basename = basename
        os.makedirs(output_dir, exist_ok=True)

        # Generated shell commands
        self.script_dir = os.path.join(self.output_dir, "scripts")
        os.makedirs(self.script_dir, exist_ok=True)

        # Base training pair
        self.base_dataset_dir = os.path.join(self.output_dir, "datasets", "base")
        self.base_style_dir = os.path.join(self.base_dataset_dir, "style")
        self.base_content_dir = os.path.join(self.base_dataset_dir, "content")
        os.makedirs(self.base_style_dir, exist_ok=True)
        os.makedirs(self.base_content_dir, exist_ok=True)

        train_image_path = Path(train_image_path)
        if not train_image_path.is_file():
            raise FileNotFoundError(f"Training image not found: {train_image_path}")

        clear_folder(self.base_style_dir)
        clear_folder(self.base_content_dir)
        with Image.open(train_image_path) as train_image:
            train_image = train_image.convert("RGB")
            train_image = train_image.resize(self.IMAGE_SIZE, Image.Resampling.LANCZOS)
            train_image.save(os.path.join(self.base_style_dir, "train_image.png"))
            
        # Checkpoints
        self.stage1_checkpoint_dir = os.path.join(self.output_dir, "checkpoints", "stage1")
        self.stage2_checkpoint_dir = os.path.join(self.output_dir, "checkpoints", "stage2")
        os.makedirs(self.stage1_checkpoint_dir, exist_ok=True)
        os.makedirs(self.stage2_checkpoint_dir, exist_ok=True)

        # Expansion training pair
        self.expand_dataset_dir = os.path.join(self.output_dir, "datasets", "expansion")
        os.makedirs(self.expand_dataset_dir, exist_ok=True)

        self.expand_style_dir = os.path.join(self.expand_dataset_dir, "style")
        self.expand_content_dir = os.path.join(self.expand_dataset_dir, "content")
        os.makedirs(self.expand_style_dir, exist_ok=True)
        os.makedirs(self.expand_content_dir, exist_ok=True)

    @staticmethod
    def validate_model_config(model_config):
        missing_keys = sorted(
            key for key in EmoLoRATrainer.REQUIRED_MODEL_CONFIG_KEYS if not model_config.get(key)
        )
        if missing_keys:
            raise ValueError(f"Missing required model config keys: {', '.join(missing_keys)}")

    def run_script(self, sh_name, sh_file_content):
        sh_file_path = os.path.join(self.script_dir, f"{sh_name}.sh")
        return create_and_run_sh_script(sh_file_path, sh_file_content)

    def stage1_checkpoint_path(self):
        return os.path.join(self.stage1_checkpoint_dir, f"checkpoint-{self.stage1_steps}")

    def ensure_directory_has_images(self, folder, label):
        images = image_files(folder)
        if not images:
            return f"Error: no images found in {label}: {folder}"
        return 0

    def build_expand_metadata(self):
        try:
            output_folder = self.expand_dataset_dir
            content_output_dir = os.path.join(output_folder, "content")
            style_output_dir = os.path.join(output_folder, "style")

            style_images = image_files(style_output_dir)

            metadata_file = os.path.join(output_folder, "metadata.jsonl")
            valid_pairs = 0
            with open(metadata_file, 'w', encoding="utf-8") as metadata:
                for image_name in style_images:
                    content_image_path = os.path.join(content_output_dir, image_name)
                    style_image_path = os.path.join(style_output_dir, image_name)

                    if not os.path.exists(content_image_path) or not os.path.exists(style_image_path):
                        print(f"Skipping incomplete expansion sample: {image_name}")
                        continue

                    json_line = {
                        "content_file_name": image_name,
                        "style_file_name": image_name,
                        "subject": subject_from_filename(image_name)
                    }
                    metadata.write(json.dumps(json_line) + '\n')
                    valid_pairs += 1

            if valid_pairs == 0:
                return "Error: no complete expansion samples are available for stage-two training"

            print(f"{valid_pairs} expansion samples and metadata were written to {output_folder}")
            return 0
        
        except Exception as e:
            return e  

    def remove_emb(self, input_folder, output_folder, sh_name):
        result = self.ensure_directory_has_images(input_folder, f"{sh_name} input")
        if result != 0:
            return result

        sh_file_content = build_remove_emb_script(
            gpu_id=self.gpu_id,
            model_config=self.model_config,
            input_folder=input_folder,
            output_folder=output_folder,
        )
        result = self.run_script(sh_name, sh_file_content)
        if result == 0:
            result = self.ensure_directory_has_images(output_folder, f"{sh_name} output")
        return result
    
    def train_stage_1(self):
        print(f"Run for <{self.basename} train_stage_1>")
        result = self.ensure_directory_has_images(self.base_content_dir, "base content")
        if result != 0:
            return result
        result = self.ensure_directory_has_images(self.base_style_dir, "base style")
        if result != 0:
            return result

        sh_file_content = build_stage1_train_script(
            gpu_id=self.gpu_id,
            model_config=self.model_config,
            stage1_steps=self.stage1_steps,
            stage1_checkpoint_dir=self.stage1_checkpoint_dir,
            base_content_dir=self.base_content_dir,
            base_style_dir=self.base_style_dir,
            style_blocks=self.STYLE_BLOCKS,
        )
        result = self.run_script("train_stage1", sh_file_content)
        if result == 0 and not os.path.isdir(self.stage1_checkpoint_path()):
            return f"Error: expected stage-one checkpoint was not created: {self.stage1_checkpoint_path()}"
        return result
    
    def write_expand_prompts(self, n=3):
        selected_prompts = random.sample(self.EXPAND_PROMPTS, min(n, len(self.EXPAND_PROMPTS)))
        output_json = os.path.join(self.expand_dataset_dir, "prompts.json")
        with open(output_json, "w", encoding="utf-8") as file:
            json.dump(selected_prompts, file, ensure_ascii=False, indent=2)
        return output_json

    def generate_expand_style_images(self, n=3):
        print(f"Run for <{self.basename} generate_expand_style_images>")
        lora_path = self.stage1_checkpoint_path()
        if not os.path.isdir(lora_path):
            return f"Error: stage-one checkpoint not found: {lora_path}"

        clear_folder(self.expand_style_dir)
        clear_folder(self.expand_content_dir)
        prompts_json = self.write_expand_prompts(n=n)
        if isinstance(prompts_json, str) and prompts_json.startswith("Error:"):
            return prompts_json

        sh_file_content = build_expand_style_images_script(
            gpu_id=self.gpu_id,
            model_config=self.model_config,
            lora_path=lora_path,
            output_folder=self.expand_style_dir,
            prompts_json=prompts_json,
        )
        result = self.run_script("generate_expand_style_images", sh_file_content)
        if result == 0:
            result = self.ensure_directory_has_images(self.expand_style_dir, "expanded style")
        return result
    
    def train_stage_2(self):
        print(f"Run for <{self.basename} train_stage_2>")  
        try:
            copy_folder_contents(
                self.stage1_checkpoint_path(),
                os.path.join(self.stage2_checkpoint_dir, f"checkpoint-{self.stage1_steps}"),
            )
        except Exception as e:
            return e

        sh_file_content = build_stage2_train_script(
            gpu_id=self.gpu_id,
            model_config=self.model_config,
            stage2_steps=self.stage2_steps,
            stage2_checkpoint_dir=self.stage2_checkpoint_dir,
            base_dataset_dir=self.base_dataset_dir,
            expand_dataset_dir=self.expand_dataset_dir,
            loss_contrastive_weight=self.LOSS_CONTRASTIVE_WEIGHT,
            loss_style_pred_expand_weight=self.LOSS_STYLE_PRED_EXPAND_WEIGHT,
            style_blocks=self.STYLE_BLOCKS,
        )
        result = self.run_script("train_stage2", sh_file_content)
        return result

    def run(self):
        # 1. Remove emb.
        result = self.remove_emb(
            input_folder=self.base_style_dir, 
            output_folder=self.base_content_dir, 
            sh_name="rem_emb"
        )
        if result != 0:
            return result

        # 2. Run stage-one training.
        result = self.train_stage_1()
        if result != 0:
            return result

        # 3. Generate complementary embroidery candidates using the stage-one EmoLoRA.
        # Current implementation:
        # - Use a lightweight variant that directly keeps n generated candidates.
        # - The following emb2des step still builds paired content images for stage-two training.
        result = self.generate_expand_style_images(n=3)
        if result != 0:
            return result

        # 4. Remove emb in the supplemental training dataset to generate content images.
        result = self.remove_emb(
            input_folder=self.expand_style_dir, 
            output_folder=self.expand_content_dir, 
            sh_name="rem_emb_expand"
        )
        if result != 0:
            return result

        # 5. Generate metadata from the content/style files in the supplemental training dataset.
        result = self.build_expand_metadata()
        if result != 0:
            return result

        # 6. Run stage-two training.
        result = self.train_stage_2()
        if result != 0:
            return result

        return result
    
    
def load_model_config(model_config_path):
    with open(model_config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a one-shot EmoLoRA from a single embroidery style image.")
    parser.add_argument(
        "--model_config",
        "--config",
        dest="model_config",
        type=str,
        required=True,
        help="Path to a JSON model config file.",
    )
    parser.add_argument("--train_image", type=str, required=True, help="Path to the reference embroidery image.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for datasets, scripts, and checkpoints.")
    parser.add_argument("--basename", type=str, default="emo_lora", help="Name used in progress logs.")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU id used for training and generation.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_config = load_model_config(args.model_config)

    trainer = EmoLoRATrainer(
        model_config=model_config,
        basename=args.basename,
        train_image_path=args.train_image,
        output_dir=args.output_dir,
        gpu_id=args.gpu_id,
    )
    result = trainer.run()
    if result != 0:
        raise RuntimeError(result)



if __name__ == "__main__":
    main()
