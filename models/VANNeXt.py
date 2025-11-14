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

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class VANNeXtBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
      
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.enhanced_lkca = EnhancedLKCA(dim)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                 requires_grad=True) if layer_scale_init_value > 0 else None

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.enhanced_mlp = EnhancedMLP(dim, mlp_hidden_dim)
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                 requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x_input = x

        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)

        attn = self.enhanced_lkca(x_norm)
        if self.gamma1 is not None:
            attn = self.gamma1.unsqueeze(-1).unsqueeze(-1) * attn

        x = x_input + self.drop_path(attn)

        x_input = x

        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)

        mlp_out = self.enhanced_mlp(x_norm)
        if self.gamma2 is not None:
            mlp_out = self.gamma2.unsqueeze(-1).unsqueeze(-1) * mlp_out

        x = x_input + self.drop_path(mlp_out)

        return x

class EnhancedLKCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv5x5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv7x7_h = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv7x7_w = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        self.spatial_fusion = nn.Conv2d(dim * 2, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        u = x.clone()

        spatial1 = self.conv5x5(x)
        spatial1 = self.act(spatial1)

        spatial2_h = self.conv7x7_h(spatial1)
        spatial2_w = self.conv7x7_w(spatial1)
        spatial2 = spatial2_h + spatial2_w

        spatial_features = torch.cat([spatial1, spatial2], dim=1)
        spatial_attn = self.spatial_fusion(spatial_features)

        channel_attn = self.channel_attention(x)

        combined_attn = spatial_attn * channel_attn
        combined_attn = torch.sigmoid(combined_attn)

        return u * combined_attn

class EnhancedMLP(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.norm = nn.LayerNorm(hidden_features, eps=1e-6)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3,
                               padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x

class VANNeXtDownsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.norm = nn.LayerNorm(in_channel, eps=1e-6)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.pw_conv = nn.Conv2d(out_channel, out_channel, 1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.pw_conv(x)
        return x

class VANNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=10,
                 depths=[3, 3, 5, 2],
                 dims=[64, 128, 256, 512],
                 mlp_ratio=4,
                 drop_path_rate=0.1,
                 layer_scale_init_value=1e-6):
        super().__init__()

        self.depths = depths
        self.stem_conv = nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=1, padding=1)
        self.stem_norm = nn.LayerNorm(dims[0] // 2, eps=1e-6)
        self.stem_conv2 = nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=1, padding=1)
        self.stem_norm2 = nn.LayerNorm(dims[0], eps=1e-6)
        self.stem_act = nn.GELU()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        cur = 0
        for i in range(len(depths)):
            blocks = []
            for j in range(depths[i]):
                blocks.append(VANNeXtBlock(
                    dim=dims[i],
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[cur + j],
                    layer_scale_init_value=layer_scale_init_value
                ))
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]

            if i < len(depths) - 1:
                self.downsample_layers.append(
                    VANNeXtDownsample(dims[i], dims[i + 1])
                )

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.stem_norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.stem_act(x)

        x = self.stem_conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.stem_norm2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.stem_act(x)

        for i in range(len(self.stages)):
            x = self.stages[i](x)
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean([-2, -1])
        x = self.norm(x)
        x = self.head(x)
        return x

def VANNeXt_Tiny(num_classes=10):
    return VANNeXt(
        depths=[3, 3, 5, 2],
        dims=[64, 128, 256, 512],
        mlp_ratio=4,
        drop_path_rate=0.1,
        num_classes=num_classes
    )

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class MixUp:
    def __init__(self, alpha=0.8):
        self.alpha = alpha

    def __call__(self, batch, labels):
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(batch.size(0))
        mixed_batch = lam * batch + (1 - lam) * batch[rand_index]
        target_a, target_b = labels, labels[rand_index]
        return mixed_batch, target_a, target_b, lam

class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch, labels):
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(batch.size(0))
        target_a = labels
        target_b = labels[rand_index]

        H, W = batch.shape[2], batch.shape[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        batch[:, :, bby1:bby2, bbx1:bbx2] = batch[rand_index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return batch, target_a, target_b, lam

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

model = VANNeXt_Tiny(num_classes=10).to(device)

print(f"VANNeXt-Tiny Parameters: {sum(p.numel() for p in model.parameters()):,}")

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

print("Training VANNeXt-Tiny for 100 epochs...")

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
        torch.save(model.state_dict(), "VANNeXt_Tiny_100epochs.pth")
        print("🔥 Saved best model")

print("Training complete!")
print(f"Best validation accuracy: {best_val_acc:.4f}")
