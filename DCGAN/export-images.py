import os
import argparse
import alive_progress

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--attack", type=str)
parser.add_argument("--batch", type=int)
parser.add_argument("--epochs", type=int)
args = parser.parse_args()

workers = 2
nc = 3
nz = 100
ngf = 64
ndf = 64
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root=f"Attacks/{args.attack}", transform=transform)

class CustomDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform:
            image = self.transform(image)
        filename = os.path.basename(path)
        class_name = self.classes[target]
        return image, filename, class_name

dataloader = DataLoader(CustomDataset(root=f"Attacks/{args.attack}", transform=transform), batch_size=1, shuffle=False)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
netG = Generator(ngpu).to(device)
netG.load_state_dict(torch.load(f'DCGAN/{args.attack}/Models/G-{args.attack}-{args.batch}-{args.epochs}.pth'))
netG.eval()

output_dir = f"DCGAN/{args.attack}/Images"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    with alive_progress.alive_bar(len(dataloader), title="[ INFO ] Generating adversarial images from DCGAN", bar='classic', spinner=None) as bar:
        for i, (real_img, filename, class_name) in enumerate(dataloader):
            real_img = real_img.to(device)

            noise = torch.randn(1, nz, 1, 1, device=device)
            fake_img = netG(noise).detach().cpu()
            fake_pil = transforms.ToPILImage()(fake_img.squeeze(0))
            class_dir = os.path.join(output_dir, class_name[0])
            os.makedirs(class_dir, exist_ok=True)
            save_path = os.path.join(class_dir, filename[0])
            fake_pil.save(save_path)
            bar()

del netG
torch.cuda.empty_cache()
torch.cuda.synchronize()