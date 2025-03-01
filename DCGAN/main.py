import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim

import utils

parser = argparse.ArgumentParser()
parser.add_argument("--attack", type=str)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--resume", type=int)
args = parser.parse_args()

device = utils.use_device()

# Set random seed for reproducibility
utils.prepare_seed(seed=16)

dataroot = f"Attacks/{args.attack}"
gpu_number = 1
image_size = 64
adam_beta = 0.5
adam_lr = 0.0002
channel_number = 3
feature_map_g = 64
feature_map_d = 64
latent_vector_z = 100
num_epochs = args.epochs
batch_size = args.batch_size

dataloader = utils.get_dataloader(dataroot, image_size=64, batch_size=batch_size, workers=4)

netG = utils.Generator(gpu_number, feature_map_g, channel_number, latent_vector_z).to(device)
netG.apply(utils.weights_init)
    
netD = utils.Discriminator(gpu_number, feature_map_d, channel_number, latent_vector_z).to(device)
netD.apply(utils.weights_init)

fixed_noise = torch.randn(64, latent_vector_z, 1, 1, device=device)

criterion = nn.BCELoss()

real_label = 1.
fake_label = 0.

optimizerD = optim.Adam(netD.parameters(), adam_lr=adam_lr, betas=(adam_beta, 0.999))
optimizerG = optim.Adam(netG.parameters(), adam_lr=adam_lr, betas=(adam_beta, 0.999))

iters = 0
G_losses = []
D_losses = []

if args.resume is None:
    start_epoch = 0
else:
    checkpoint = torch.load(f'DCGAN/{args.attack}/Models/Checkpoint-{args.attack}-Epoch-{args.resume}-{batch_size}.pth')
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    G_losses.append(checkpoint['G_loss'])
    D_losses.append(checkpoint['D_loss'])
    start_epoch = checkpoint['epoch'] + 1

for epoch in range(start_epoch, num_epochs):
    for i, data in enumerate(dataloader, 0):

        # Update D network
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        noise = torch.randn(b_size, latent_vector_z, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Update G network
        netG.zero_grad()
        label.fill_(real_label)  
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        iters += 1

        if epoch % 500 == 0 or epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'G_loss': G_losses,
                'D_loss': D_losses
            }, f'DCGAN/{args.attack}/Models/Checkpoint-{args.attack}-Epoch-{epoch}-{batch_size}.pth')

torch.save(netG.state_dict(), f'DCGAN/{args.attack}/Models/G-{args.attack}-{batch_size}-{num_epochs}.pth')

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'DCGAN/FGSM/Loss-{batch_size}-{num_epochs}.pdf')

del netD
del netG
torch.cuda.empty_cache()
torch.cuda.synchronize()