import utils
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--attack", type=str)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
args = parser.parse_args()

dataroot = "Dataset/Imagewoof"
gpu_number = 1
adam_beta = 0.5
adam_lr = 0.0002
feature_map_e = 64
feature_map_g = 64
channel_number = 3
latent_vector_z = 100
num_epochs = args.epochs
batch_size = args.batch_size

device = utils.use_device()

dataloader = utils.get_dataloader(dataroot, image_size=64, batch_size=batch_size, workers=4)

netE = utils.Encoder(feature_map_e, channel_number, latent_vector_z).to(device)

netG = utils.Generator(gpu_number, feature_map_g, channel_number, latent_vector_z).to(device)
netG.apply(utils.weights_init)

checkpoint = torch.load(f"DCGAN/{args.attack}/Models/Checkpoint-{args.attack}-Epoch-{num_epochs}-{batch_size}.pth", weights_only=True)
netG.load_state_dict(checkpoint["netG_state_dict"])
netG.to(device)
netG.eval()

criterion = nn.MSELoss()

# vgg_loss_fn = utils.VGGPerceptualLoss(layer=8).to(device)
optimizerE = optim.Adam(netE.parameters(), lr=adam_lr, betas=(adam_beta, 0.999))

num_epochs = 100
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        optimizerE.zero_grad()

        latent_vectors = netE(real_images)
        latent_vectors = latent_vectors.view(real_images.size(0), 100, 1, 1)

        reconstructed_images = netG(latent_vectors)

        loss = criterion(reconstructed_images, real_images)

        loss.backward()
        optimizerE.step()

        if i % 100 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] Loss: {loss.item():.4f}")

torch.save(netE.state_dict(), f"DCGAN/{args.attack}/Models/Encoder-Final.pth")