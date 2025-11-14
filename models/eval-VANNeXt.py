print("\n🚀 Starting VANNeXt-Tiny Comprehensive Evaluation...")

def evaluate_vannext_comprehensive(model_path="VANNeXt_Tiny_100epochs.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    model = VANNeXt_Tiny(num_classes=10).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Loaded VANNeXt-Tiny model from {model_path}")

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in tqdm(testloader, desc="Evaluating on test set"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)

            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_acc = correct / total

    print(f"\n{'='*60}")
    print(f"📊 VANNeXt-Tiny EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc:.4f} ({correct}/{total})")
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.8})
    plt.title('VANNeXt-Tiny - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Labels', fontsize=12, fontweight='bold')
    plt.ylabel('True Labels', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('VANNeXt_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
  
    print(f"\n📈 Classification Report:")
    print("=" * 60)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    print(f"\n🎯 Per-class Accuracy Analysis:")
    print("=" * 60)
    class_correct = cm.diagonal()
    class_total = cm.sum(axis=1)
    class_acc = class_correct / class_total

    sorted_indices = np.argsort(class_acc)[::-1]

    for i in sorted_indices:
        acc = class_acc[i]
        class_name = class_names[i]
        print(f"  {class_name:12s}: {acc:.4f} ({class_correct[i]:4d}/{class_total[i]:4d}) {'✅' if acc > 0.9 else '⚠️ ' if acc > 0.8 else '❌'}")

    print(f"\n📊 Overall Statistics:")
    print("=" * 60)
    print(f"Best Class Accuracy: {class_acc.max():.4f} ({class_names[class_acc.argmax()]})")
    print(f"Worst Class Accuracy: {class_acc.min():.4f} ({class_names[class_acc.argmin()]})")
    print(f"Average Class Accuracy: {class_acc.mean():.4f}")
    print(f"Std of Class Accuracy: {class_acc.std():.4f}")

    return test_acc, cm, all_preds, all_labels, all_probs

def analyze_vannext_predictions(model_path="VANNeXt_Tiny_100epochs.pth", num_samples=12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    model = VANNeXt_Tiny(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct_samples = []
    incorrect_samples = []

    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            probs = torch.softmax(outputs, dim=1)

            for i in range(len(imgs)):
                if preds[i] == labels[i] and len(correct_samples) < num_samples:
                    correct_samples.append({
                        'image': imgs[i].cpu(),
                        'true_label': labels[i].item(),
                        'pred_label': preds[i].item(),
                        'confidence': probs[i][preds[i]].item(),
                        'all_probs': probs[i].cpu().numpy()
                    })
                elif preds[i] != labels[i] and len(incorrect_samples) < num_samples:
                    incorrect_samples.append({
                        'image': imgs[i].cpu(),
                        'true_label': labels[i].item(),
                        'pred_label': preds[i].item(),
                        'confidence': probs[i][preds[i]].item(),
                        'all_probs': probs[i].cpu().numpy()
                    })

                if len(correct_samples) >= num_samples and len(incorrect_samples) >= num_samples:
                    break
            if len(correct_samples) >= num_samples and len(incorrect_samples) >= num_samples:
                break

    def display_samples(samples, title, num_cols=4):
        num_samples = len(samples)
        num_rows = (num_samples + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4*num_rows))
        if num_rows == 1:
            axes = [axes] if num_cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, sample in enumerate(samples):
            if i >= len(axes):
                break

            img = sample['image'].permute(1, 2, 0)
            img = img * torch.tensor([0.2023, 0.1994, 0.2010]) + torch.tensor([0.4914, 0.4822, 0.4465])
            img = torch.clamp(img, 0, 1)

            axes[i].imshow(img)
            true_name = class_names[sample['true_label']]
            pred_name = class_names[sample['pred_label']]
            conf = sample['confidence']

            color = 'green' if sample['true_label'] == sample['pred_label'] else 'red'
            status = "✓" if sample['true_label'] == sample['pred_label'] else "✗"

            axes[i].set_title(f"{status} True: {true_name}\nPred: {pred_name}\nConf: {conf:.3f}",
                            color=color, fontsize=10, fontweight='bold')
            axes[i].axis('off')

        for i in range(len(samples), len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'VANNeXt-Tiny - {title}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    if correct_samples:
        display_samples(correct_samples, f"Correct Predictions (Confidence: {np.mean([s['confidence'] for s in correct_samples]):.3f})")

    if incorrect_samples:
        display_samples(incorrect_samples, f"Incorrect Predictions (Confidence: {np.mean([s['confidence'] for s in incorrect_samples]):.3f})")

    print(f"\n🔍 VANNeXt Prediction Analysis:")
    print("=" * 50)
    print(f"Correct predictions analyzed: {len(correct_samples)}")
    print(f"Incorrect predictions analyzed: {len(incorrect_samples)}")

    if correct_samples:
        avg_conf_correct = np.mean([s['confidence'] for s in correct_samples])
        print(f"Average confidence on correct: {avg_conf_correct:.3f}")

    if incorrect_samples:
        avg_conf_incorrect = np.mean([s['confidence'] for s in incorrect_samples])
        print(f"Average confidence on incorrect: {avg_conf_incorrect:.3f}")

def vannext_confidence_analysis(model_path="VANNeXt_Tiny_100epochs.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    model = VANNeXt_Tiny(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    confidences_correct = []
    confidences_incorrect = []

    with torch.no_grad():
        for imgs, labels in tqdm(testloader, desc="Analyzing confidence"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            max_probs = probs.max(1)[0]

            for i in range(len(imgs)):
                if preds[i] == labels[i]:
                    confidences_correct.append(max_probs[i].item())
                else:
                    confidences_incorrect.append(max_probs[i].item())

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(confidences_correct, bins=50, alpha=0.7, color='green', label='Correct')
    plt.hist(confidences_incorrect, bins=50, alpha=0.7, color='red', label='Incorrect')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('VANNeXt-Tiny - Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.boxplot([confidences_correct, confidences_incorrect],
                labels=['Correct', 'Incorrect'])
    plt.title('VANNeXt-Tiny - Confidence Box Plot')
    plt.ylabel('Confidence')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('VANNeXt_confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n📊 VANNeXt Confidence Analysis:")
    print("=" * 40)
    print(f"Correct predictions: {len(confidences_correct)}")
    print(f"Incorrect predictions: {len(confidences_incorrect)}")
    print(f"Mean confidence (correct): {np.mean(confidences_correct):.4f}")
    print(f"Mean confidence (incorrect): {np.mean(confidences_incorrect):.4f}")
    print(f"Std confidence (correct): {np.std(confidences_correct):.4f}")
    print(f"Std confidence (incorrect): {np.std(confidences_incorrect):.4f}")

test_acc, cm, all_preds, all_labels, all_probs = evaluate_vannext_comprehensive()

analyze_vannext_predictions()

vannext_confidence_analysis()

print(f"\n🎉 VANNeXt-Tiny Evaluation Complete!")
print(f"📈 Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

with open("VANNeXt_results.txt", "w") as f:
    f.write(f"VANNeXt-Tiny Evaluation Results\n")
    f.write("=" * 50 + "\n")
    f.write(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
    f.write(f"Confusion Matrix:\n{cm}\n")
    f.write("\nPer-class accuracy:\n")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
    class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_acc):
        f.write(f"{class_names[i]}: {acc:.4f}\n")

print(f"\n💾 Results saved to 'VANNeXt_results.txt'")
