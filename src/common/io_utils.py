import os
import shutil


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def clear_folder(folder):
    os.makedirs(folder, exist_ok=True)
    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def image_files(folder):
    if not os.path.isdir(folder):
        return []

    return sorted(
        file_name for file_name in os.listdir(folder)
        if (
            os.path.isfile(os.path.join(folder, file_name))
            and os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS
        )
    )


def subject_from_filename(image_name):
    return image_name.rsplit("_", 1)[0].replace("_", " ")


def copy_folder_contents(source_folder, target_folder):
    if not os.path.isdir(source_folder):
        raise FileNotFoundError(f"Source folder not found: {source_folder}")

    os.makedirs(target_folder, exist_ok=True)
    clear_folder(target_folder)
    shutil.copytree(source_folder, target_folder, dirs_exist_ok=True)
    print(f"Copied checkpoint from {source_folder} to {target_folder}")
