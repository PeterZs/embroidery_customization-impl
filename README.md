<h2 align="center">
  One-shot Embroidery Customization <br> via Contrastive LoRA Modulation
</h2>

<p align="center">
  <a href="">
    <img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'>
  </a>
  <a href="https://style3d.github.io/embroidery_customization">
    <img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'>
  </a>
  <a href="https://huggingface.co/Style3D/EmoLoRA">
    <img src="https://img.shields.io/badge/HuggingFace-EmoLoRA-yellow?logo=huggingface&logoColor=white" alt="HuggingFace Model">
  </a>
</p>

<p align="center">
  <img src="assets/images/representative_image.jpg" width="100%">
</p>


> Diffusion models have significantly advanced image manipulation techniques, and their ability to generate photorealistic images is beginning to transform retail workflows, particularly in presale visualization. Beyond artistic style transfer, the capability to perform fine-grained visual feature transfer is becoming increasingly important.
> Embroidery is a textile art form characterized by intricate interplay of diverse stitch patterns and material properties, which poses unique challenges for existing style transfer methods. To explore the customization for such fine-grained features, we propose a novel contrastive learning framework that disentangles fine-grained style and content features with a single reference image, building on the classic concept of image analogy. We first construct an image pair to define the target style, and then adopt a similarity metric based on the decoupled representations of pretrained diffusion models for style-content separation. Subsequently, we propose a two-stage contrastive LoRA modulation technique to capture fine-grained style features. In the first stage, we iteratively update the whole LoRA and the selected style blocks to initially separate style from content. In the second stage, we design a contrastive learning strategy to further decouple style and content through self-knowledge distillation. Finally, we build an inference pipeline to handle image or text inputs with only the style blocks.
> To evaluate our method on fine-grained style transfer, we build a benchmark for embroidery customization. Our approach surpasses prior methods on this task and further demonstrates strong generalization to three additional domains: artistic style transfer, sketch colorization, and appearance transfer.

## Getting Started

### Installation

1. Clone the repository

```
git clone https://github.com/Style3D/embroidery_customization-impl.git
cd embroidery_customization-impl
```

2. Create and activate a virtual environment (venv)

```
python3 -m venv venv
source venv/bin/activate
```

> On Windows:

```
venv\Scripts\activate
```

3. Install dependencies

```
pip install -r requirements.txt
```

### Training

#### Model Config

Before training, edit `configs/model_config.json` and set the model paths for your machine. Most entries can be either a HuggingFace model id or a local directory path.

```json
{
  "sdxl_path": "stabilityai/stable-diffusion-xl-base-1.0",
  "vae_path": "madebyollin/sdxl-vae-fp16-fix",
  "controlnet_canny_path": "xinsir/controlnet-canny-sdxl-1.0",
  "controlnet_tile_path": "xinsir/controlnet-tile-sdxl-1.0",
  "sd3_path": "stabilityai/stable-diffusion-3-medium-diffusers",
  "sd3_controlnet_canny_path": "InstantX/SD3-Controlnet-Canny",
  "sd3_controlnet_tile_path": "InstantX/SD3-Controlnet-Tile",
  "wd14_models_dir": "/path/to/wd14",
  "hed_model_dir": "lllyasviel/Annotators"
}
```

You can also download these HuggingFace models locally and replace the corresponding values in `configs/model_config.json` with local paths.

Additional model setup:

- `wd14_models_dir`: download `SmilingWolf/wd-convnext-tagger-v3/model.onnx` and `SmilingWolf/wd-convnext-tagger-v3/selected_tags.csv`; rename them to `wd-convnext-tagger-v3.onnx` and `wd-convnext-tagger-v3.csv`, then place both files in the same `wd14` folder.
- `hed_model_dir`: download `lllyasviel/Annotators/ControlNetHED.pth` and place it in an `Annotators` folder. The folder path should be used as `hed_model_dir`.

#### Train

Train an EmoLoRA from a single embroidery reference image:

```bash
python src/emo_lora_trainer.py \
  --model_config configs/model_config.json \
  --train_image path/to/reference_embroidery_image.png \
  --output_dir path/to/output_dir \
  --gpu_id 0
```

### Pretrained Model

We release the pretrained EmoLoRA model on HuggingFace:

👉 **HuggingFace Model:**  
https://huggingface.co/Style3D/EmoLoRA

### Inference

We provide an example inference workflow in `inference.ipynb`.
Open the notebook and run the cells to reproduce the example results.

```
jupyter notebook inference.ipynb
```
