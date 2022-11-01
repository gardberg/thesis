from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch
from torch.optim import Adam

import matplotlib.pyplot as plt
import diff_unet
from diff_unet import train, Unet, linear_beta_schedule, cosine_beta_schedule
from plot_utils import row_plot

import wandb
wandb.init(project="mnist_diff_test")
do_log = True

batch_size = 1
subs = 1

torch.manual_seed(0)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = MNIST('./data', train=True, download=True, transform=transform)

# Create dataset of 10 images
mnist = torch.utils.data.Subset(mnist, range(subs))
data_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)



img = next(iter(data_loader))[0]
T = 200
t = torch.randint(0, T, (batch_size,))

batch_size, channels, height, width = img.shape

def scaled_cosine(x):
    return cosine_beta_schedule(x) * 0.3

unet = Unet(dim=height,
            channels=channels,
            dim_mults=(1,2,4),
            resnet_block_groups=7,
            use_convnext=False,
            timesteps=T,
            schedule_f=linear_beta_schedule,)

if do_log: wandb.watch(unet, log="all", log_freq=10)

optimizer = Adam(unet.parameters(), lr=1e-3)


train(unet, optimizer, data_loader, epochs=800, wandb_log=do_log)


ims = diff_unet.sample(unet, height, batch_size=batch_size)


step_size = T // 10
ims10 = ims[0:-1:step_size]
row_plot([im.view(1, height, width) for im in ims10])

# Save last tensor in ims as image
# plt.imsave('mnist_training_test.png', ims[-1].view(height, width), cmap='gray')