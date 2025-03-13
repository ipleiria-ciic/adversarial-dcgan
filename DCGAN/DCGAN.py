import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim

import Utils

parser = argparse.ArgumentParser()
parser.add_argument("--attack", type=str)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--delta", type=str)
parser.add_argument("--resume", type=int)
args = parser.parse_args()

device = Utils.use_device()

# Set random seed for reproducibility
Utils.prepare_seed(seed=16)

dataroot = f"Attacks/{args.attack}-{args.delta}"
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

dataloader = Utils.get_dataloader(dataroot, image_size=64, batch_size=batch_size, workers=4)

netG = Utils.Generator(gpu_number, feature_map_g, channel_number, latent_vector_z).to(device)
netG.apply(Utils.weights_init)
    
netD = Utils.Discriminator(gpu_number, feature_map_d, channel_number, latent_vector_z).to(device)
netD.apply(Utils.weights_init)

fixed_noise = torch.randn(64, latent_vector_z, 1, 1, device=device)

criterion = nn.BCELoss()

real_label = 1.
fake_label = 0.

optimizerD = optim.Adam(netD.parameters(), lr=adam_lr, betas=(0.0001, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=adam_lr, betas=(0.0002, 0.999))

schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerD, mode='min', factor=0.8, patience=30, min_lr=1e-6)
schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG, mode='min', factor=0.8, patience=30, min_lr=1e-6)

iters = 0
G_losses = []
D_losses = []
best_errG = float('inf')

if args.resume == 0 or args.resume is None:
    start_epoch = 0
else:
    checkpoint = torch.load(f'DCGAN/{args.attack}/Models/Checkpoint-{args.attack}-Epoch-{args.resume}-{batch_size}.pth')
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    schedulerD.load_state_dict(checkpoint['schedulerD_state_dict'])
    schedulerG.load_state_dict(checkpoint['schedulerG_state_dict'])
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
            debug_lr_D = optimizerD.param_groups[0]['lr']
            debug_lr_G = optimizerG.param_groups[0]['lr']

            print(f'[Epoch {epoch}/{num_epochs}][Batch {i:02}/{len(dataloader)}] - Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f} | LR_D: {debug_lr_D:.4f} | LR_G: {debug_lr_G:.4f}')

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iters += 1

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'schedulerG_state_dict': schedulerG.state_dict(),
                'schedulerD_state_dict': schedulerD.state_dict(),
                'G_loss': G_losses,
                'D_loss': D_losses
            }, f'DCGAN/{args.attack}/Models/Checkpoint-{args.attack}-Delta-{args.delta}-Epoch-{epoch}-{batch_size}.pth')

        if errG.item() < best_errG:
            best_errG = errG.item()
            torch.save({
                'epoch': epoch,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'schedulerG_state_dict': schedulerG.state_dict(),
                'schedulerD_state_dict': schedulerD.state_dict(),
                'G_loss': G_losses,
                'D_loss': D_losses
            }, f'DCGAN/{args.attack}/Models/Best-Checkpoint-{args.attack}-Delta-{args.delta}-Epoch-{epoch}-{batch_size}.pth')
    
    schedulerG.step(errG.item())
    schedulerD.step(errD.item())

torch.save({
    'netG_state_dict': netG.state_dict(),
    'netD_state_dict': netD.state_dict(),
    'optimizerG_state_dict': optimizerG.state_dict(),
    'optimizerD_state_dict': optimizerD.state_dict(),
    'schedulerG_state_dict': schedulerG.state_dict(),
    'schedulerD_state_dict': schedulerD.state_dict(),
    'G_loss': G_losses,
    'D_loss': D_losses
}, f'DCGAN/{args.attack}/Models/Checkpoint-{args.attack}-Delta-{args.delta}-{batch_size}.pth')

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'DCGAN/{args.attack}/Loss-{batch_size}-Delta-{args.delta}-{num_epochs}.pdf')

del netD
del netG
torch.cuda.empty_cache()
torch.cuda.synchronize()