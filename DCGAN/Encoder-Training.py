import os
import Utils
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--attack", type=str)
parser.add_argument("--checkpoint_epochs", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--delta", type=str)
args = parser.parse_args()

dataroot = "Dataset/Imagewoof"
gpu_number = 1
adam_beta = 0.5
adam_lr = 0.0002
feature_map_e = 64
feature_map_g = 64
channel_number = 3
latent_vector_z = 100
best_epoch = args.checkpoint_epochs
attack = args.attack
num_epochs = args.epochs
batch_size = args.batch_size
delta = float(args.delta)

dir = f"DCGAN/{args.attack}/Models/Encoder-Delta-{delta:.02f}"
os.makedirs(dir, exist_ok=True)

device = Utils.use_device()

dataloader = Utils.get_dataloader(dataroot, image_size=64, batch_size=batch_size, workers=4)

netE = Utils.Encoder(feature_map_e, channel_number, latent_vector_z).to(device)

netG = Utils.Generator(gpu_number, feature_map_g, channel_number, latent_vector_z).to(device)
netG.apply(Utils.weights_init)

checkpoint_name = f"DCGAN/{attack}/Models/Delta-{delta:.02f}/Best-Checkpoint-{attack}-Delta-{delta:.02f}-Epoch-{best_epoch}-{batch_size}.pth"
checkpoint = torch.load(checkpoint_name, weights_only=True)
netG.load_state_dict(checkpoint["netG_state_dict"])
netG.to(device)
netG.eval()

criterion = nn.MSELoss()

# vgg_loss_fn = Utils.VGGPerceptualLoss(layer=8).to(device)
optimizerE = optim.Adam(netE.parameters(), lr=adam_lr, betas=(adam_beta, 0.999))

encoder_loss = []
best_loss = float('inf')

print(f"[ INFO ] Encoder training for {attack} ({delta})")

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
            print(f"[Epoch {epoch}/{num_epochs}][Batch {i:03}/{len(dataloader)}] - Loss: {loss.item():.4f}")
        
        encoder_loss.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                'epoch': epoch,
                'netE_state_dict': netE.state_dict(),
                'optimizerE_state_dict': optimizerE.state_dict(),
                'best_loss': best_loss,
                'encoder_losses': encoder_loss
            }, f"DCGAN/{args.attack}/Models/Encoder-Delta-{delta:.02f}/Best-Checkpoint-Encoder-{batch_size}.pth")

torch.save({
    'netE_state_dict': netE.state_dict(),
    'optimizerE_state_dict': optimizerE.state_dict(),
    'encoder_losses': encoder_loss
}, f"DCGAN/{args.attack}/Models/Encoder-Delta-{delta:.02f}/Encoder.pth")

del netG, netE
torch.cuda.empty_cache() 
torch.cuda.synchronize()