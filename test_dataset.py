from datasets import load_dataset

print("Loading PlantVillage images...")

ds = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0"
)

print("\nDataset loaded successfully!")
print(ds)

print("\nAvailable columns:")
print(ds["train"].column_names)

print("\nFirst sample:")
sample = ds["train"][0]

print("Image:", sample["image"])
print("Label:", sample["class_label"])