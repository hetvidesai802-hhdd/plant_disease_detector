
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import shutil

DATA_DIR = Path("data")

def main():

    print("Loading PlantVillage dataset...")

    ds = load_dataset(
        "geraldmc/plantvillage-full",
        revision="v0.1.0"
    )["train"]

    print("Dataset loaded successfully!")
    print("Total images:", len(ds))

    # Extract labels and original split information
    labels = ds["class_label"]
    splits = ds["split"]

    class_names = sorted(set(labels))

    print("Number of classes:", len(class_names))

    # Separate original training and testing indices
    train_indices = [
        i for i, s in enumerate(splits)
        if s == "train"
    ]

    test_indices = [
        i for i, s in enumerate(splits)
        if s == "test"
    ]

    # Create validation data from training data
    train_labels = [labels[i] for i in train_indices]

    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=0.10,
        random_state=42,
        stratify=train_labels
    )

    # Remove old generated folders, if present
    #for folder in ["train", "val", "test"]:
    #    path = DATA_DIR / folder

    #    if path.exists():
    #        shutil.rmtree(path)

    # Prepare all three datasets
    splits_to_create = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices
    }

    for split_name, indices in splits_to_create.items():

        print(f"\nPreparing {split_name}: {len(indices)} images")

        for i in tqdm(indices):

            sample = ds[i]
            class_name = sample["class_label"]
            image = sample["image"].convert("RGB")

            target_dir = DATA_DIR / split_name / class_name

            target_dir.mkdir(parents=True, exist_ok=True)

            image_path = target_dir / f"image_{i:06d}.jpg"

            if image_path.exists():
                continue

            try:
                image.save(image_path, quality=95)

            except PermissionError:
                print(f"\nPermission denied: {image_path}")
                raise

    # Save class names
    (DATA_DIR / "class_names.json").write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8"
    )

    print("\nDataset preparation completed!")

    print("Training images:", len(train_indices))
    print("Validation images:", len(val_indices))
    print("Testing images:", len(test_indices))

    print("Dataset location:", DATA_DIR.resolve())


if __name__ == "__main__":
    main()
