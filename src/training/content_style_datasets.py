from torch.utils.data import Dataset
import torch
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms
import os
import json

def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a list of dictionaries.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list: A list of dictionaries with the data from the file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse each line as JSON and append to the list
            data.append(json.loads(line.strip()))
    return data


class PairDataset(Dataset):
    def __init__(
        self,
        instance_data_root_content,
        instance_data_root_style,
        instance_prompt_content,
        instance_prompt_style,
        size=1024,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_prompt_content = instance_prompt_content
        self.instance_prompt_style = instance_prompt_style

        self.instance_data_root_content = Path(instance_data_root_content)
        if not self.instance_data_root_content.exists():
            raise ValueError("Instance images content root doesn't exists.")
        
        self.instance_data_root_style = Path(instance_data_root_style)
        if not self.instance_data_root_style.exists():
            raise ValueError("Instance images style root doesn't exists.")

        instance_images_content = [Image.open(path) for path in list(Path(instance_data_root_content).iterdir())]
        instance_images_style = [Image.open(path) for path in list(Path(instance_data_root_style).iterdir())]

        # assert len(instance_images_content) == len(instance_images_style) == 1, "Only one image per folder is currently supported."
        assert len(instance_images_content) == len(instance_images_style), "Only one image per folder is currently supported."

        self.instance_images_content = instance_images_content
        self.instance_images_style = instance_images_style

        # image processing to prepare for using SD-XL micro-conditioning
        self.original_sizes_content = [(content.height,content.width) for content in self.instance_images_content]
        self.original_sizes_style = [ (style.height,style.width)for style in self.instance_images_style]


        self.image_transforms = transforms.Compose(
            [
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.num_instance_images = len(self.instance_images_content)

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        if isinstance(index, list):
            index = index[0]  # Handle list indices
        index = index % self.num_instance_images  # Wrap index to prevent overflow

        example = {}
        # Load and preprocess content image
        instance_image_content = self.instance_images_content[index]
        instance_image_content = exif_transpose(instance_image_content)
        if not instance_image_content.mode == "RGB":
            instance_image_content = instance_image_content.convert("RGB")

        # Load and preprocess style image
        instance_image_style = self.instance_images_style[index]
        instance_image_style = exif_transpose(instance_image_style)
        if not instance_image_style.mode == "RGB":
            instance_image_style = instance_image_style.convert("RGB")

        # Resize and transform images
        resize_transform = transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR)
        instance_image_content = resize_transform(instance_image_content)
        instance_image_style = resize_transform(instance_image_style)

        instance_image_content_torch = transforms.ToTensor()(instance_image_content).to(dtype=torch.float32)
        instance_image_style_torch = transforms.ToTensor()(instance_image_style).to(dtype=torch.float32)

        # Stack and apply transformations
        instance_image_batch = torch.stack([instance_image_content_torch, instance_image_style_torch])
        instance_image_batch = self.image_transforms(instance_image_batch)
        instance_image_content, instance_image_style = instance_image_batch[0], instance_image_batch[1]

        # Original sizes
        original_size_content = self.original_sizes_content[index]
        original_size_style = self.original_sizes_style[index]

        # Build example dictionary
        example["instance_images_content"] = instance_image_content
        example["instance_images_style"] = instance_image_style
        example["instance_prompt_content"] = self.instance_prompt_content
        example["instance_prompt_style"] = self.instance_prompt_style
        example["original_size_content"] = original_size_content
        example["original_size_style"] = original_size_style
        example["crop_top_left"] = (0, 0)

        return example


def collate_fn(examples):
    if not examples:
        raise ValueError("Received an empty batch in collate_fn.")
    pixel_values_content = [example["instance_images_content"] for example in examples]
    pixel_values_style = [example["instance_images_style"] for example in examples]
    prompts_content = [example["instance_prompt_content"] for example in examples]
    prompts_style = [example["instance_prompt_style"] for example in examples]
    original_sizes_content = [example["original_size_content"] for example in examples]
    original_sizes_style = [example["original_size_style"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]

    pixel_values_content = torch.stack(pixel_values_content)
    pixel_values_content = pixel_values_content.to(memory_format=torch.contiguous_format).float()

    pixel_values_style = torch.stack(pixel_values_style)
    pixel_values_style = pixel_values_style.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values_content": pixel_values_content,
        "pixel_values_style": pixel_values_style,
        "prompts_content": prompts_content,
        "prompts_style": prompts_style,
        "original_sizes_content": original_sizes_content,
        "original_sizes_style": original_sizes_style,
        "crop_top_lefts": crop_top_lefts,
    }
    return batch



class MixedPairDataset(Dataset):
    def __init__(
        self,
        train_data_dir_base,
        train_data_dir_expand,
        content_prompt_base,        
        size=1024,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop

        instance_data_root_content_base = os.path.join(train_data_dir_base, "content")   
        instance_data_root_style_base = os.path.join(train_data_dir_base, "style")

        instance_images_content_base = [Image.open(path) for path in list(Path(instance_data_root_content_base).iterdir())]
        instance_images_style_base = [Image.open(path) for path in list(Path(instance_data_root_style_base).iterdir())]
        
        self.instance_images_content_prompts_base = content_prompt_base
        self.instance_images_content_base = instance_images_content_base
        self.instance_images_style_base = instance_images_style_base


        instance_data_root_content = os.path.join(train_data_dir_expand, "content")   
        instance_data_root_style = os.path.join(train_data_dir_expand, "style")
        metadata_path = os.path.join(train_data_dir_expand, "metadata.jsonl")

        instance_images_content = []
        instance_images_style = []
        instance_images_content_prompts = []

        data = read_jsonl(metadata_path)
        for entry in data:
            content_file = entry.get("content_file_name")
            style_file = entry.get("style_file_name")
            subject = entry.get("subject")
            instance_images_content.append(Image.open(os.path.join(instance_data_root_content, content_file)))
            instance_images_style.append(Image.open(os.path.join(instance_data_root_style, style_file)))
            instance_images_content_prompts.append(subject)

        assert len(instance_images_content) == len(instance_images_style) == len(instance_images_content_prompts) 


        self.instance_images_content = instance_images_content
        self.instance_images_style = instance_images_style
        self.instance_images_content_prompts = instance_images_content_prompts

        original_sizes_content=[]
        original_sizes_style=[]
        for i in range(len(self.instance_images_content)):
            original_sizes_content.append((self.instance_images_content[i].height, self.instance_images_content[i].width))
            original_sizes_style.append((self.instance_images_style[i].height, self.instance_images_style[i].width))

        self.original_sizes_content = original_sizes_content
        self.original_sizes_style = original_sizes_style

        self.image_transforms = transforms.Compose(
            [
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.num_instance_images = len(self.instance_images_content)

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        # print(f"Fetching index: {index}, dataset length: {self.num_instance_images}")
        if isinstance(index, list):
            index = index[0]  # Handle list indices
        index = index % self.num_instance_images  # Wrap index to prevent overflow
        # print(f"Using wrapped index: {index}")

        example = {}
        #############################
        # Base train data
        instance_image_content_base = self.instance_images_content_base[0]
        instance_image_content_base = exif_transpose(instance_image_content_base)
        if not instance_image_content_base.mode == "RGB":
            instance_image_content_base = instance_image_content_base.convert("RGB")

        # Load and preprocess style image
        instance_image_style_base = self.instance_images_style_base[0]
        instance_image_style_base = exif_transpose(instance_image_style_base)
        if not instance_image_style_base.mode == "RGB":
            instance_image_style_base = instance_image_style_base.convert("RGB")

        # Resize and transform images
        resize_transform = transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR)
        instance_image_content_base = resize_transform(instance_image_content_base)
        instance_image_style_base = resize_transform(instance_image_style_base)

        instance_image_content_torch_base = transforms.ToTensor()(instance_image_content_base).to(dtype=torch.float32)
        instance_image_style_torch_base = transforms.ToTensor()(instance_image_style_base).to(dtype=torch.float32)

        # Stack and apply transformations
        instance_image_batch_base = torch.stack([instance_image_content_torch_base, instance_image_style_torch_base])
        instance_image_batch_base = self.image_transforms(instance_image_batch_base)
        instance_image_content_base, instance_image_style_base = instance_image_batch_base[0], instance_image_batch_base[1]

        # prompt
        instance_prompt_content_base = self.instance_images_content_prompts_base
        instance_prompt_style_base = instance_prompt_content_base + ' in [emb] style'        

        # Load and preprocess content image
        instance_image_content = self.instance_images_content[index]
        instance_image_content = exif_transpose(instance_image_content)
        if not instance_image_content.mode == "RGB":
            instance_image_content = instance_image_content.convert("RGB")

        # Load and preprocess style image
        instance_image_style = self.instance_images_style[index]
        instance_image_style = exif_transpose(instance_image_style)
        if not instance_image_style.mode == "RGB":
            instance_image_style = instance_image_style.convert("RGB")

        # Resize and transform images
        resize_transform = transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR)
        instance_image_content = resize_transform(instance_image_content)
        instance_image_style = resize_transform(instance_image_style)

        instance_image_content_torch = transforms.ToTensor()(instance_image_content).to(dtype=torch.float32)
        instance_image_style_torch = transforms.ToTensor()(instance_image_style).to(dtype=torch.float32)

        # Stack and apply transformations
        instance_image_batch = torch.stack([instance_image_content_torch, instance_image_style_torch])
        instance_image_batch = self.image_transforms(instance_image_batch)
        instance_image_content, instance_image_style = instance_image_batch[0], instance_image_batch[1]

        # prompt
        instance_prompt_content = self.instance_images_content_prompts[index]
        instance_prompt_style = instance_prompt_content + ' in [emb] style'
        

        # Original sizes
        original_size_content = self.original_sizes_content[index]
        original_size_style = self.original_sizes_style[index]

        # Build example dictionary
        example["instance_images_content_base"] = instance_image_content_base
        example["instance_images_style_base"] = instance_image_style_base
        example["instance_prompt_content_base"] = instance_prompt_content_base
        example["instance_prompt_style_base"] = instance_prompt_style_base        
        example["instance_images_content"] = instance_image_content
        example["instance_images_style"] = instance_image_style
        example["instance_prompt_content"] = instance_prompt_content
        example["instance_prompt_style"] = instance_prompt_style
        example["original_size_content"] = original_size_content
        example["original_size_style"] = original_size_style
        example["crop_top_left"] = (0, 0)

        return example


def mixed_collate_fn(examples):
    if not examples:
        raise ValueError("Received an empty batch in collate_fn.")
        
    # Base data
    instance_images_content_base = examples[0]["instance_images_content_base"]
    instance_images_style_base = examples[0]["instance_images_style_base"]
    instance_prompt_content_base = examples[0]["instance_prompt_content_base"]
    instance_prompt_style_base = examples[0]["instance_prompt_style_base"]

    # Add base as the first element
    pixel_values_content = [instance_images_content_base] + [example["instance_images_content"] for example in examples]
    pixel_values_style = [instance_images_style_base] + [example["instance_images_style"] for example in examples]
    prompts_content = [instance_prompt_content_base] + [example["instance_prompt_content"] for example in examples]
    prompts_style = [instance_prompt_style_base] + [example["instance_prompt_style"] for example in examples]
    original_sizes_content = [examples[0]["original_size_content"]] + [example["original_size_content"] for example in examples]
    original_sizes_style = [examples[0]["original_size_style"]] + [example["original_size_style"] for example in examples]
    crop_top_lefts = [examples[0]["crop_top_left"]] + [example["crop_top_left"] for example in examples]

    # Stack and format tensors
    pixel_values_content = torch.stack(pixel_values_content)
    pixel_values_content = pixel_values_content.to(memory_format=torch.contiguous_format).float()

    pixel_values_style = torch.stack(pixel_values_style)
    pixel_values_style = pixel_values_style.to(memory_format=torch.contiguous_format).float()

    # Prepare the final batch
    batch = {
        "pixel_values_content": pixel_values_content,
        "pixel_values_style": pixel_values_style,
        "prompts_content": prompts_content,
        "prompts_style": prompts_style,
        "original_sizes_content": original_sizes_content,
        "original_sizes_style": original_sizes_style,
        "crop_top_lefts": crop_top_lefts,
    }
    return batch
