import sys
import torch
import torchvision
import torch.nn as nn
import torchvision.utils as vutils
from torchvision import datasets, transforms
from network import Generator, Discriminator

# Hyper parameter setting
batch_size = 64
learning_rate = 2e-4
weight_decay = 1e-7
train_epoch = 10000


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(28),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist', train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(
    G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerD = torch.optim.SGD(
    D.parameters(), lr=1e-3, weight_decay=1e-7, momentum=0)

label = torch.FloatTensor(batch_size)
real_label, fake_label = 1, 0

for epoch in range(train_epoch):
    for i, (imgs, _) in  enumerate(dataloader):
        # fix G, train D
        optimizerD.zero_grad()
        # load imgs to cuda
        imgs = imgs.to(device)
        # Let D output to 1
        output = D(imgs)
        label.data.fill_(real_label)
        label = label.to(device)
        output = torch.max(output, dim=1).values
        output = torch.clamp(output, min=0)
        # print(label.size())
        # print(output.size())
        lossD_real = criterion(output, label)
        lossD_real.backward()
        label.data.fill_(fake_label)
        noise = torch.randn(batch_size, 100, 1, 1).to(device)
        # Make fake imgs
        fake = G(noise)
        output = D(fake.detach())
        output = torch.max(output, dim=1).values
        output = torch.clamp(output, min=0)
        lossD_fake = criterion(output, label)
        lossD_fake.backward()
        lossD = lossD_fake + lossD_real
        optimizerD.step()
        # Fix D, train G
        optimizerG.zero_grad()
        # Let D to 1
        label.data.fill_(real_label)
        label = label.to(device)
        output = D(fake)
        output = torch.max(output, dim=1).values
        output = torch.clamp(output, min=0)
        # print(output.size())
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()
        # Show loss
        if i%100 == 0:
            loss_st = "[{}/{}] Loss D: {:.6f}, Loss G: {:.6f}"
            loss_st = loss_st.format(epoch+1, train_epoch, lossD.item(), lossG.item())
            print(loss_st, end="\r")
    print()
    if (epoch + 1) % 10 == 0:
        vutils.save_image(
            fake.data,
            "./fakepng/fake_epoch{}.png".format(epoch+1)
        )
        torch.save(G.state_dict(), "./gan/G_{:05d}.pth".format(epoch+1))
        torch.save(D.state_dict(), "./gan/D_{:05d}.pth".format(epoch+1))
