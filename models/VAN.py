import os, math, random
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

seed = 42
random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

full_train = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
testset = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

train_size = 45000
val_size = 5000
trainset, valset = random_split(full_train, [train_size, val_size])
valset.dataset.transform = test_tf  

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
valloader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

model = van_b0_exact(num_classes=10).to(device)
print(f"VAN-B0 Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=0.05, betas=(0.9, 0.999))

def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=100)

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

mixup = MixUp(alpha=0.8)
cutmix = CutMix(alpha=1.0)

best_val_acc = 0

print("Training VAN-B0 with EXACT SAME data as VANNeXt...")

for epoch in range(100):
    model.train()
    running_loss = 0

    for imgs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/100"):
        imgs, labels = imgs.to(device), labels.to(device)

        if np.random.random() < 0.5:
            mixed_imgs, target_a, target_b, lam = cutmix(imgs, labels)
            optimizer.zero_grad()
            outputs = model(mixed_imgs)
            loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
        else:
            mixed_imgs, target_a, target_b, lam = mixup(imgs, labels)
            optimizer.zero_grad()
            outputs = model(mixed_imgs)
            loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in valloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1} | Loss: {running_loss/len(trainloader):.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "VAN_B0_EXACT_SAME_DATA.pth")
        print("🔥 Saved best model")

print("Training complete!")
print(f"Best validation accuracy: {best_val_acc:.4f}")
