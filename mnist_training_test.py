from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch
import matplotlib.pyplot as plt
import diff_unet
# reload diff_unet package
import importlib
importlib.reload(diff_unet)

batch_size = 1
subs = 1

# torch.manual_seed(0)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = MNIST('./data', train=True, download=True, transform=transform)

# Create dataset of 10 images
mnist = torch.utils.data.Subset(mnist, range(subs))
data_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)



from diff_unet import Unet, linear_beta_schedule, cosine_beta_schedule

img = next(iter(data_loader))[0]
T = 400
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
            schedule_f=scaled_cosine,)

from torch.optim import Adam

optimizer = Adam(unet.parameters(), lr=1e-3)

from diff_unet import train
train(unet, optimizer, data_loader, epochs=2000)


ims = diff_unet.sample(unet, height, batch_size=batch_size)
from plot_utils import row_plot

step_size = T // 10
ims10 = ims[0:-1:step_size]
row_plot([im.view(1, height, width) for im in ims10])

# Save last tensor in ims as image
plt.imsave('mnist_training_test.png', ims[-1].view(height, width), cmap='gray')