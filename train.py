import argparse
import csv
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


# ==============================================================================
# Dataset — builds train / val / test splits from class subfolders
# ==============================================================================

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def build_splits(root_dir: str, n_test: int = 5, seed: int = 42):

    root = Path(root_dir)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    trainval_samples = []
    test_samples     = []

    for cls in classes:
        cls_dir = root / cls
         # leave images for testing
        imgs = sorted([
            str(f) for f in cls_dir.iterdir()
            if f.suffix.lower() in IMG_EXTENSIONS
        ])
        label   = class_to_idx[cls]

        if len(imgs) < 30:
            effective_n_test = 1
        else:
            effective_n_test = min(n_test, max(1, len(imgs) // 5))
        cutoff  = max(0, len(imgs) - effective_n_test)
        trainval_samples += [(p, label) for p in imgs[:cutoff]]
        test_samples     += [(p, label) for p in imgs[cutoff:]]

    rng = random.Random(seed)
    rng.shuffle(trainval_samples)

    n_val          = max(1, int(len(trainval_samples) * 0.2))
    n_train        = len(trainval_samples) - n_val
    train_samples  = trainval_samples[:n_train]
    val_samples    = trainval_samples[n_train:]

    return classes, class_to_idx, train_samples, val_samples, test_samples


class EmotionDataset(Dataset):

    def __init__(self, samples, transform=None, cache=True):
        self.samples   = samples
        self.transform = transform
        self.cache     = cache
        self._cache_d  = {} 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        if idx in self._cache_d:
            img = self._cache_d[idx]
        else:
            img = Image.open(path).convert("RGB")
            if self.cache:
                self._cache_d[idx] = img

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_weights(self, num_classes: int) -> torch.Tensor:
       # weights smaller classes so they are slighlty overfit and can train
        counts = torch.zeros(num_classes)
        for _, label in self.samples:
            counts[label] += 1
        counts   = torch.clamp(counts, min=1) 
        weights  = 1.0 / counts
        return torch.tensor([weights[label].item() for _, label in self.samples])


## EfficientNet-B0 with custom classification head

def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """
    Loads pretrained EfficientNet-B0 and replaces the final layer
    with a new Linear layer sized for num_classes emotions.

    freeze_backbone=True  → only the new head trains (faster, less accurate)
    freeze_backbone=False → entire network fine-tunes (default, recommended)
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the ImageNet head (1280 → 1000) with our head (1280 → num_classes)
    in_features = model.classifier[1].in_features   # 1280 for EfficientNet-B0
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    """One full pass through the training data. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch [{epoch}/{total_epochs}] "
                  f"Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}")

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device, return_preds=False):
    """
    Runs inference on loader without updating weights.
    Returns (avg_loss, accuracy, per_class_correct, per_class_total).
    If return_preds=True, also returns (all_true, all_pred) lists for confusion matrix.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    per_class_correct, per_class_total = {}, {}
    all_true, all_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs   = model(images)
            loss      = criterion(outputs, labels)
            predicted = outputs.argmax(dim=1)

            total_loss += loss.item()
            correct    += (predicted == labels).sum().item()
            total      += labels.size(0)

            for label, pred in zip(labels, predicted):
                l = label.item()
                per_class_correct[l] = per_class_correct.get(l, 0) + (pred == label).item()
                per_class_total[l]   = per_class_total.get(l, 0) + 1
                all_true.append(l)
                all_pred.append(pred.item())

    if return_preds:
        return total_loss / len(loader), correct / total, per_class_correct, per_class_total, all_true, all_pred
    return total_loss / len(loader), correct / total, per_class_correct, per_class_total


def print_per_class(classes, per_class_correct, per_class_total):
    """Prints per-class accuracy breakdown."""
    for idx, name in enumerate(classes):
        c   = per_class_correct.get(idx, 0)
        t   = per_class_total.get(idx, 0)
        pct = (c / t * 100) if t > 0 else 0
        print(f"    {name:<14}: {c}/{t} = {pct:.0f}%")



# MAIN TRAINFING LOOP
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Building dataset splits...")
    classes, class_to_idx, train_samples, val_samples, test_samples = build_splits(
        root_dir = args.data_dir,
        n_test   = args.n_test,
    )
    num_classes = len(classes)

    print(f"Classes ({num_classes}): {classes}")
    print(f"Train: {len(train_samples)}  |  Val: {len(val_samples)}  |  Test (held out): {len(test_samples)}")

    # Show per-class test holdout counts
    print("\nTest holdout per class:")
    from collections import Counter
    test_counts = Counter(label for _, label in test_samples)
    for idx, name in enumerate(classes):
        print(f"  {name:<14}: {test_counts.get(idx, 0)} images held out")

    train_ds = EmotionDataset(train_samples, transform=train_tf, cache=args.cache)
    val_ds   = EmotionDataset(val_samples,   transform=eval_tf,  cache=args.cache)
    test_ds  = EmotionDataset(test_samples,  transform=eval_tf,  cache=True)

    # adds weight for underrepressented classes
    sample_weights = train_ds.get_class_weights(num_classes)
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )

    # DataLoaders
    nw = 0 if os.name == "nt" else 4

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        sampler     = sampler,
        num_workers = nw,
        pin_memory  = (device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = nw,
        pin_memory  = (device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = nw,
        pin_memory  = (device.type == "cuda"),
    )

    # the model
    print(f"\nBuilding EfficientNet-B0 for {num_classes} classes...")
    model = build_model(num_classes=num_classes, freeze_backbone=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # --- Checkpoint directory and log ---
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = checkpoint_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                 "val_loss", "val_acc", "lr"])

    #  training loop
    best_val_acc = 0.0
    print(f"\nStarting training for {args.epochs} epochs...\n")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs
        )
        val_loss, val_acc, per_cls_c, per_cls_t = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.0f}s)")
        print(f"  Train — Loss: {train_loss:.4f}  Acc: {train_acc*100:.1f}%")
        print(f"  Val   — Loss: {val_loss:.4f}  Acc: {val_acc*100:.1f}%")
        print(f"  LR: {current_lr:.2e}")

        if epoch % 5 == 0:
            print("  Per-class val accuracy:")
            print_per_class(classes, per_cls_c, per_cls_t)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc"        : val_acc,
                "classes"        : classes,
            }, checkpoint_dir / "best_model.pt")
            print(f"  *** New best model saved ({val_acc*100:.1f}%)")

        # Save latest checkpoint (for resuming if job times out)
        torch.save({
            "epoch"          : epoch,
            "model_state"    : model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc"        : val_acc,
            "classes"        : classes,
        }, checkpoint_dir / "last_model.pt")

        # Append to CSV log
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.4f}", f"{train_acc:.4f}",
                f"{val_loss:.4f}",   f"{val_acc:.4f}",
                f"{current_lr:.2e}",
            ])

        print("=" * 60)

    # test set evaluation (new images)
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION (never seen during training)")
    print("=" * 60)

    # Load the best model weights for test evaluation
    ckpt = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, test_acc, test_cls_c, test_cls_t, all_true, all_pred = evaluate(
        model, test_loader, criterion, device, return_preds=True
    )

    print(f"Overall test accuracy: {test_acc*100:.1f}%  ({int(test_acc*len(test_samples))}/{len(test_samples)})")
    print("\nPer-class test accuracy:")
    print_per_class(classes, test_cls_c, test_cls_t)

    # Save confusion matrix as CSV
    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(all_true, all_pred):
        cm[t][p] += 1

    cm_path = checkpoint_dir / "confusion_matrix.csv"
    with open(cm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + classes)
        for i, row in enumerate(cm):
            writer.writerow([classes[i]] + row)
    print(f"\nConfusion matrix saved to: {cm_path}")

    print(f"\nTraining complete.")
    print(f"Best val accuracy:  {best_val_acc*100:.1f}%")
    print(f"Test accuracy:      {test_acc*100:.1f}%")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Training log: {log_path}")


# ==============================================================================
# Command-line interface
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dog emotion classifier")

    parser.add_argument("--data_dir",       default="./data/raw/DogEmotion",
                        help="Path to dataset root with class subfolders")
    parser.add_argument("--checkpoint_dir", default="./checkpoints",
                        help="Where to save model weights and training log")
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--n_test",         type=int,   default=5,
                        help="Number of images per class to hold out as test set")
    parser.add_argument("--no_rembg",       action="store_true",
                        help="Skip background removal (use when data is pre-processed)")
    parser.add_argument("--no_cache",       action="store_true",
                        help="Don't cache images in RAM")

    args = parser.parse_args()
    args.cache = not args.no_cache

    train(args)