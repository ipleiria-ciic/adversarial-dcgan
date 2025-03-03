import os
import scipy
import lpips
import torch
import warnings
import torchvision
import numpy as np
import alive_progress

class Generator(torch.nn.Module):
    """
    Generator class.
    """
    def __init__(self, gpu_number, feature_map_g, channel_number, latent_vector_z):
        super(Generator, self).__init__()
        self.gpu_number = gpu_number
        self.feature_map_g = feature_map_g
        self.channel_number = channel_number
        self.latent_vector_z = latent_vector_z
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(latent_vector_z, feature_map_g*8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(feature_map_g*8),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(feature_map_g*8, feature_map_g * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(feature_map_g*4),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(feature_map_g*4, feature_map_g*2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(feature_map_g*2),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(feature_map_g*2, feature_map_g, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(feature_map_g),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(feature_map_g, channel_number, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class InceptionV3FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        self.inception = torchvision.models.inception_v3(weights='DEFAULT', transform_input=False)
        self.inception.fc = torch.nn.Identity()

    def forward(self, x):
        return self.inception(x)
    
def use_device():
    """
    Gets the device available to use while using Torch.

    Return:
    - device: Device (CPU or GPU).
    """
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return device

def fetch_checkpoint(path, device):
    """
    Fetchs a given checkpoint.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    return checkpoint

def transform_images():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def custom_dataloader(attack, batch_size=1):
    """
    A custom Dataloader to load the images generated from the DCGAN.
    """
    class CustomDataset(torchvision.datasets.ImageFolder):
        def __getitem__(self, index):
            path, target = self.samples[index]
            image = self.loader(path)
            if self.transform:
                image = self.transform(image)
            filename = os.path.basename(path)
            class_name = self.classes[target]
            return image, filename, class_name
        
    transform = transform_images()

    dataloader = torch.utils.data.DataLoader(CustomDataset(root=f"../Attacks/{attack}", transform=transform), batch_size=batch_size, shuffle=True)

    return dataloader

def generate_images(dataloader, iteration, netG, device):
    output_dir = f"Generated-Images-{iteration}"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        with alive_progress.alive_bar(len(dataloader), title=f"[ INFO ] Generating adversarial images", bar='classic', spinner=None) as bar:
            for i, (real_img, filename, class_name) in enumerate(dataloader):
                real_img = real_img.to(device)
                noise = torch.randn(1, 100, 1, 1, device=device)
                
                fake_img = netG(noise).detach().cpu()

                fake_pil = torchvision.transforms.ToPILImage()(fake_img.squeeze(0))

                class_dir = os.path.join(output_dir, class_name[0])
                os.makedirs(class_dir, exist_ok=True)

                save_path = os.path.join(class_dir, filename[0]) 
                fake_pil.save(save_path)

                bar()

def load_model(name, device):
    model = torch.jit.load(f'../Models/{name}.pt')
    model.eval()
    model.to(device)
    return model

def load_dataset(path):
    transform = transform_images()
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    return dataloader

def classify_images(model, dataloader, device, title):
    predictions = {}
    with torch.no_grad():
        with alive_progress.alive_bar(len(dataloader), title=f"[ INFO ] {title}", bar='classic', spinner=None) as bar:
            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                for idx, (pred, label) in enumerate(zip(preds.cpu().numpy(), labels.cpu().numpy())):
                    predictions[i * dataloader.batch_size + idx] = (pred, label)
                bar()
    return predictions

def calculate_lpips(original_loader, adversarial_loader, device):
    warnings.filterwarnings("ignore", module="torch")
    warnings.filterwarnings("ignore", module="lpips")

    lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
    lpips_values = []
    with alive_progress.alive_bar(len(original_loader), title=f"[ INFO ] LPIPS similarity calculation", bar='classic', spinner=None) as bar:
        for (img1, _), (img2, _) in zip(original_loader, adversarial_loader):
            img1, img2 = img1.to(device), img2.to(device)
            lpips_value = lpips_fn(img1, img2).mean().item()
            lpips_values.append(lpips_value)
            bar()
    return np.mean(lpips_values)

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
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

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
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch 
            
                images = images.to(device)
                activations = model(images)
                all_activations.append(activations.cpu())
                bar()
    torch_activations = torch.cat(all_activations, dim=0).numpy()
    torch.cuda.empty_cache()
    return torch_activations

def fid(real_dataset_path, generated_dataset_path, device):
    """
    Preparation of both real and generated datasets for the FID calculation.

    Parameters:
    - real_dataset_path: Real dataset path.
    - generated_dataset_path: Generated (adversarial) dataset path.
    """

    model = InceptionV3FeatureExtractor().eval()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((299, 299)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    real_dataset = torchvision.datasets.ImageFolder(real_dataset_path, transform=transform)
    generated_dataset = torchvision.datasets.ImageFolder(generated_dataset_path, transform=transform)
    
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=8, shuffle=False, num_workers=4)
    generated_loader = torch.utils.data.DataLoader(generated_dataset, batch_size=8, shuffle=False, num_workers=4)

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