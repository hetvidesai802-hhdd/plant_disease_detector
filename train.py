"""Fine-tune ImageNet-pretrained ResNet18 on the prepared PlantVillage folders."""
from pathlib import Path
import json
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 16
EPOCHS = 3
IMAGE_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    if not (DATA_DIR / "train").exists():
        raise FileNotFoundError("Run `python prepare_data.py` first.")

    weights = ResNet18_Weights.DEFAULT
    mean, std = weights.transforms().mean, weights.transforms().std
    tfms = {
        "train": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ]),
        "val": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ]),
        "test": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ]),
    }
    datasets_by_split = {
        s: datasets.ImageFolder(DATA_DIR / s, transform=tfms[s])
        for s in ("train", "val", "test")
    }
    class_names = datasets_by_split["train"].classes
    for split in ("val", "test"):
        if datasets_by_split[split].classes != class_names:
            raise ValueError(f"Class folders differ in {split} split.")

    loaders = {
        s: DataLoader(datasets_by_split[s], batch_size=BATCH_SIZE,
                      shuffle=(s == "train"), num_workers=0,
                      pin_memory=(DEVICE.type == "cuda"))
        for s in ("train", "val", "test")
    }

    print(f"Device: {DEVICE} | Classes: {len(class_names)}")
    model = models.resnet18(weights=weights)
    # Keep early generic visual features; fine-tune layer4 and the new classifier.
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)
    best_acc, best_weights = 0.0, copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        for phase in ("train", "val"):
            model.train(phase == "train")
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in loaders[phase]:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
            loss_avg, acc = running_loss / total, correct / total
            print(f"{phase}: loss={loss_avg:.4f} accuracy={acc:.4%}")
            if phase == "val" and acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())
        scheduler.step()

    model.load_state_dict(best_weights)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loaders[phase], desc=phase):
            outputs = model(images.to(DEVICE))
            correct += (outputs.argmax(1).cpu() == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    print(f"\nHeld-out test accuracy: {test_acc:.4%}")

    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "mean": mean,
        "std": std,
        "test_accuracy": test_acc,
    }, MODEL_DIR / "plant_disease_resnet18.pth")
    print("Saved models/plant_disease_resnet18.pth")

if __name__ == "__main__":
    main()
