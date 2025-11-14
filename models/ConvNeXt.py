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

class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvNeXtBlock_Exact(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class ConvNeXt_Exact_CIFAR(nn.Module):
    def __init__(self, in_chans=3, num_classes=10, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0.1, layer_scale_init_value=1e-6, head_init_scale=1.0):
        super().__init__()

        self.stem = nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2) 
        self.stem_norm = LayerNorm2d(dims[0], eps=1e-6)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        curr_dim = dims[0]
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock_Exact(dim=curr_dim, drop_path=dpr[sum(depths[:i]) + j],
                                  layer_scale_init_value=layer_scale_init_value)
                for j in range(depths[i])
            ])
            self.stages.append(stage)

            if i < 3: 
                downsample = nn.Sequential(
                    LayerNorm2d(curr_dim, eps=1e-6),
                    nn.Conv2d(curr_dim, dims[i+1], kernel_size=2, stride=2),
                )
                self.downsample_layers.append(downsample)
                curr_dim = dims[i+1]

        self.norm = LayerNorm2d(curr_dim, eps=1e-6)
        self.head = nn.Linear(curr_dim, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x) 
        x = self.stem_norm(x)

        for i in range(4):
            x = self.stages[i](x)
            if i < 3: 
                x = self.downsample_layers[i](x) 

        return self.norm(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean([-2, -1]) 
        x = self.head(x)
        return x

def convnext_exact_tiny(num_classes=10):
    return ConvNeXt_Exact_CIFAR(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.1,
        num_classes=num_classes
    )

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
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

model = convnext_exact_tiny(num_classes=10).to(device)
print(f"ConvNeXt-Exact Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=4e-3,
    weight_decay=0.05,
    betas=(0.9, 0.999)
)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

epochs = 100
warmup_epochs = 5
total_steps = epochs * len(trainloader)
warmup_steps = warmup_epochs * len(trainloader)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_val_acc = 0
print("🚀 Training ConvNeXt-Exact with CORRECT CIFAR setup...")

for epoch in range(epochs):
    model.train()
    running_loss = 0

    pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{scheduler.get_last_lr()[0]:.6f}'})

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in valloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_acc = 100. * correct / total
    avg_loss = running_loss / len(trainloader)

    print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "ConvNeXt_Exact_Tiny_CIFAR.pth")
        print(f"🔥 New best model saved! Val Acc: {val_acc:.2f}%")

print(f"🎯 Training completed! Best validation accuracy: {best_val_acc:.2f}%")
