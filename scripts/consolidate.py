import os
import shutil
from pathlib import Path

from tqdm import tqdm


def consolidate_ragtruth_explicit_no_prefix():
    # Define source folders and target folder
    source_folders = [
        "/app/cache/mtopdiv/ragtruth_qa",
        "/app/cache/mtopdiv/ragtruth_summ",
        "/app/cache/mtopdiv/ragtruth_data2txt",
    ]
    target_root = "/app/cache/mtopdiv/ragtruth"
    model_names = [
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf",
        "Mistral-7B-Instruct-v0.1",
    ]

    # Create target root directory
    os.makedirs(target_root, exist_ok=True)

    file_count = 0

    for model in model_names:
        for source_folder in tqdm(source_folders):
            source_base = Path(source_folder) / "zero_out_prompt" / model

            if not source_base.exists():
                print(f"Warning: {source_base} not found, skipping...")
                continue

            # Find all layer_i/head_j directories
            for layer_dir in tqdm(source_base.glob("layer_*")):
                if not layer_dir.is_dir():
                    continue

                for head_dir in layer_dir.glob("head_*"):
                    if not head_dir.is_dir():
                        continue

                    # Build target path
                    target_path = (
                        Path(target_root)
                        / "zero_out_prompt"
                        / model
                        / layer_dir.name
                        / head_dir.name
                    )
                    target_path.mkdir(parents=True, exist_ok=True)

                    # Copy all files from source to target
                    for file in head_dir.iterdir():
                        if file.is_file():
                            target_file = target_path / file.name

                            shutil.copy2(file, target_file)

    print(f"\nConsolidation complete! Processed {file_count} files.")
    print(f"All files are now in the '{target_root}' folder.")


if __name__ == "__main__":
    consolidate_ragtruth_explicit_no_prefix()
