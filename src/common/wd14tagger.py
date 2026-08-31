import os
import csv
import numpy as np
from PIL import Image
from onnxruntime import InferenceSession


class WD14Tagger:
    def __init__(self, models_dir, ort_providers=None):
        self.models_dir = models_dir
        self.ort_providers = ort_providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._model_cache = {}
        self._tags_cache = {}

    def _load_model(self, model_name):
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        model_path = os.path.join(self.models_dir, f"{model_name}.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name}.onnx not found in {self.models_dir}")

        model = InferenceSession(model_path, providers=self.ort_providers)
        self._model_cache[model_name] = model
        return model

    def _load_tags(self, model_name, replace_underscore=True):
        cache_key = (model_name, replace_underscore)
        if cache_key in self._tags_cache:
            return self._tags_cache[cache_key]

        csv_path = os.path.join(self.models_dir, f"{model_name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file for model {model_name} not found in {self.models_dir}")
        
        tags = []
        general_index = None
        character_index = None

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row.
            for row in reader:
                if general_index is None and row[2] == "0":
                    general_index = reader.line_num - 2
                elif character_index is None and row[2] == "4":
                    character_index = reader.line_num - 2
                tag_name = row[1].replace("_", " ") if replace_underscore else row[1]
                tags.append(tag_name)

        result = tags, general_index, character_index
        self._tags_cache[cache_key] = result
        return result

    def _preprocess_image(self, image, target_size):
        ratio = float(target_size) / max(image.size)
        new_size = tuple([int(x * ratio) for x in image.size])
        image = image.resize(new_size, Image.LANCZOS)
        square = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        square.paste(image, ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2))

        image = np.array(square).astype(np.float32)
        image = image[:, :, ::-1]  # RGB -> BGR
        image = np.expand_dims(image, 0)  # Add the batch dimension.
        return image

    def tag_image(self, image, model_name, threshold=0.35, character_threshold=0.85, 
                  exclude_tags="", replace_underscore=True, trailing_comma=False):
        model = self._load_model(model_name)
        tags, general_index, character_index = self._load_tags(model_name, replace_underscore)

        input_tensor = model.get_inputs()[0]
        target_size = input_tensor.shape[1]

        image_data = self._preprocess_image(image, target_size)

        label_name = model.get_outputs()[0].name
        probs = model.run([label_name], {input_tensor.name: image_data})[0]

        result = list(zip(tags, probs[0]))
        general = [item for item in result[general_index:character_index] if item[1] > threshold]
        character = [item for item in result[character_index:] if item[1] > character_threshold]

        all_tags = character + general

        exclude_list = [tag.strip().lower() for tag in exclude_tags.split(",") if tag.strip()]
        filtered_tags = [tag for tag in all_tags if tag[0].lower() not in exclude_list]

        res = ("" if trailing_comma else ", ").join(
            item[0].replace("(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "")
            for item in filtered_tags
        )

        return res

    def tag_batch(self, images, model_name, **kwargs):
        results = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            result = self.tag_image(img, model_name, **kwargs)
            results.append(result)
        return results
