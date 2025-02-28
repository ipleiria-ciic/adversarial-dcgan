import os
import lpips
import warnings
import numpy as np
import alive_progress

from datetime import datetime
from scipy.linalg import sqrtm
from PIL import Image, ImageChops
from statistics import mean, harmonic_mean, median

import torch
import torch.nn as nn

from torchvision import models, transforms
from torchvision.transforms import v2 as T
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('jpg', 'png', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        self.inception = models.inception_v3(weights='DEFAULT', transform_input=False)
        self.inception.fc = nn.Identity()

    def forward(self, x):
        return self.inception(x)
    
def use_device():
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return device

def get_activations(data_loader, model, device, title):
    """
    Get the activations for an entire Dataloader.

    Parameters:
    - data_loader: A given Dataloader.
    - model: Model to use.
    - device: Device to use (CPU | GPU).

    Return:
    - torch_activations: Activations in a Torch Tensor format.
    """

    model = model.to(device)
    all_activations = []
    with torch.no_grad():
        with alive_progress.alive_bar(len(data_loader), title=f"[ INFO ] {title}", bar='classic', spinner=None) as bar:
            for images in data_loader:
                images = images.to(device)
                activations = model(images)
                all_activations.append(activations.cpu())
                bar()
    torch_activations = torch.cat(all_activations, dim=0).numpy()
    torch.cuda.empty_cache()
    return torch_activations


def calculate_fid(mu1, sigma1, mu2, sigma2):
    """
    Calculates the FID score between two Gaussian distributions.

    Parameters:
    - mu1: First mean (usually the original one).
    - sigma1: First covariance (usually the original one).
    - mu2: Second mean (usually the adversarial one).
    - sigma2: Second covariance (usually the adversarial one).

    Return:
    - fid: FID score.
    """

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def fid(real_dataset_path, generated_dataset_path):
    """
    Preparation of both real and generated datasets for the FID calculation.

    Parameters:
    - real_dataset_path: Real dataset path.
    - generated_dataset_path: Generated (adversarial) dataset path.
    """

    model = InceptionV3FeatureExtractor().eval()
    device = use_device()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    real_dataset = ImageDataset(image_dir=real_dataset_path, transform=transform)
    generated_dataset = ImageDataset(image_dir=generated_dataset_path, transform=transform)

    real_loader = DataLoader(real_dataset, batch_size=8, shuffle=False, num_workers=4)
    generated_loader = DataLoader(generated_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Calculate activations for real and generated images.
    real_activations = get_activations(real_loader, model, device, title="Calculating the activations of the real images")
    generated_activations = get_activations(generated_loader, model, device, title="Calculating the activations of the generated images")

    # Calculate mean and covariance.
    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    mu_gen = np.mean(generated_activations, axis=0)
    sigma_gen = np.cov(generated_activations, rowvar=False)

    # Compute FID.
    fid_score = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)

    return fid_score

def calculate_lpips(real_dataset_path, generated_dataset_path, network='vgg', aggregation='mean'):
    """
    Calculate LPIPS similarity for the entire dataset.

    Parameters:
    - real_dataset_path: Real dataset path.
    - generated_dataset_path: Generated (adversarial) dataset path.
    - network: Network to use in the LPIPS calculation. Default: VGG.
    - aggregation: Aggregation method for final similarity. Options: 'mean', 'harmonic_mean', 'median'. Default: 'mean'.
    """

    # Disable the warning from torch (built-in in LPIPS).
    warnings.filterwarnings("ignore", module="torch")
    warnings.filterwarnings("ignore", module="lpips")

    loss_fn = lpips.LPIPS(net=network, verbose=False)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    real_images = sorted(os.listdir(real_dataset_path))
    generated_images = sorted(os.listdir(generated_dataset_path))

    # Align dataset sizes: the size must the same.
    if len(real_images) > len(generated_images):
        real_images = real_images[:len(generated_images)]
    else:
        generated_images = generated_images[:len(real_images)]

    # Ensure both datasets have the same number of images.
    assert len(real_images) == len(generated_images), "Datasets must have the same number of images."

    # Calculate LPIPS for each image pair.
    lpips_scores = []
    with alive_progress.alive_bar(len(real_images), title=f"[ INFO ] LPIPS similarity calculation", bar='classic', spinner=None) as bar:
        for real_img_name, gen_img_name in zip(real_images, generated_images):
            real_img_path = os.path.join(real_dataset_path, real_img_name)
            gen_img_path = os.path.join(generated_dataset_path, gen_img_name)

            real_img = transform(Image.open(real_img_path).convert("RGB")).unsqueeze(0)
            gen_img = transform(Image.open(gen_img_path).convert("RGB")).unsqueeze(0)

            # Calculate LPIPS similarity.
            similarity = loss_fn(real_img, gen_img).item()
            lpips_scores.append(similarity)

            bar()

    if aggregation == 'mean':
        final_score = mean(lpips_scores)
    elif aggregation == 'harmonic_mean':
        final_score = harmonic_mean(lpips_scores)
    elif aggregation == 'median':
        final_score = median(lpips_scores)
    else:
        raise ValueError("[ ERROR ] Invalid aggregation method. Choose from 'mean', 'harmonic_mean', or 'median'.")

    # print(f"[ \033[92mRESULTS\033[0m ] Final LPIPS similarity: {final_score:.02f}. Aggregation used: '{aggregation}'.")
    return final_score, aggregation