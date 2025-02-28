import argparse
import alive_progress

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import Utils

parser = argparse.ArgumentParser()
parser.add_argument("--attack", type=str)
parser.add_argument("--model", type=str)
args = parser.parse_args()

device = Utils.use_device()

model = torch.jit.load(f'Models/{args.model}.pt')
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

original_dataset = ImageFolder("Dataset/Imagewoof/train", transform=transform)
adversarial_dataset = ImageFolder(f"DCGAN/{args.attack}/Images", transform=transform)

original_loader = DataLoader(original_dataset, batch_size=8, shuffle=False)
adversarial_loader = DataLoader(adversarial_dataset, batch_size=8, shuffle=False)

def classify_images(model, dataloader, title):
    predictions = {}
    with torch.no_grad():
        with alive_progress.alive_bar(len(dataloader), title=f"[ INFO ] {title}", bar='classic', spinner=None) as bar:
            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                for idx, (pred, label) in enumerate(zip(preds.cpu().numpy(), labels.cpu().numpy())):
                    predictions[i * dataloader.batch_size + idx] = (pred, label)
    return predictions

# Original Classification
orig_preds = classify_images(model, original_loader, title="Original image classification")
correctly_classified = {k: v[0] for k, v in orig_preds.items() if v[0] == v[1]}
print(f"[ \033[92mRESULTS\033[0m ] Correctly classified original images: {len(correctly_classified)}")

# Adversarial Classification -> Fooling Rate
adv_preds = classify_images(model, adversarial_loader, title="Adversarial image classification")
fooling_count = sum(1 for k, v in correctly_classified.items() if k in adv_preds and adv_preds[k][0] != v)
fooling_rate = fooling_count / len(correctly_classified) if correctly_classified else 0
print(f"[ \033[92mRESULTS\033[0m ] Fooling Rate (FR): {100*fooling_rate:.2f}%")

# FID Classification
fid_score = Utils.fid(real_dataset_path="Dataset/Imagewoof/train", generated_dataset_path=f"DCGAN/{args.attack}/Images")
print(f"[ \033[92mRESULTS\033[0m ] FID Score: {fid_score:.2f}")

# LPIPS Classification
lpips_score = Utils.calculate_lpips(real_dataset_path="Dataset/Imagewoof/train", generated_dataset_path=f"DCGAN/{args.attack}/Images")
print(f"[ \033[92mRESULTS\033[0m ] LPIPS Score: {lpips_score:.4f}")