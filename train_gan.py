import torch
import torch.nn as nn
import numpy as np
from glob import glob
import scipy.io as sio
import matplotlib.pyplot as plt
from train_utils import Generator, Discriminator, SLFDataset

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
now = datetime.now().strftime("%y_%m_%d__%H_%M_%S")
writer = SummaryWriter("logs/"+now)


def gen_cmap(data):
    plt.imshow( data[0].cpu().numpy(), cmap="jet" )
    return plt.gcf()


def train(dataloader, dim_gauss, G,  D, n_epochs, lr, n_critic, beta1, device):
    loss_func = nn.BCELoss()
    optimizer_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1/2, 0.999))
    # optimizer_d = torch.optim.RMSprop(D.parameters(), lr=lr)
    # optimizer_g = torch.optim.RMSprop(G.parameters(), lr=lr)
    
    # print("optimizer for Discriminator: ", optimizer_d)
    # print("optimizer for Generator: ", optimizer_g)
    G.to(device)
    D.to(device)
    loss_all_d = []
    loss_all_g = []
    itrs = 0
    for epoch in range(n_epochs):
        loss_d_real_epoch = []
        loss_d_fake_epoch = []
        loss_d_epoch = []
        loss_g_epoch = []

        noise_factor = 1.0 - (epoch / n_epochs)
        for batch_idx, images in enumerate(dataloader):
            
            batch_size = len(images)
            labels_true = torch.ones(batch_size, device=device)
            labels_false = torch.zeros(batch_size, device=device)
            labels_true -= (torch.rand( batch_size ) * 0.15 * noise_factor).to(device)
            labels_false += (torch.rand(batch_size ) * 0.15 * noise_factor).to(device)

            images = images.to(device)
            z = torch.randn(batch_size, 1, dim_gauss).to(device)
            
            for i in range(n_critic):
                optimizer_d.zero_grad()
                D_x = D(images).squeeze()
                loss_d_real = loss_func(D_x, labels_true)

                D_G_z1 = D(G(z)).squeeze()
                loss_d_fake = loss_func(D_G_z1, labels_false)
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimizer_d.step()

            
            optimizer_g.zero_grad()
            fake = G(z)
            D_G_z2 = D(fake).squeeze()
            loss_g = loss_func(D_G_z2, labels_true)
            loss_g.backward()
            optimizer_g.step()

            loss_d_real_epoch.append(loss_d_real.item())
            loss_d_fake_epoch.append(loss_d_fake.item())
            loss_d_epoch.append(loss_d.item())
            loss_g_epoch.append(loss_g.item())

            with torch.no_grad():
                # Output training stats
                if batch_idx % 10 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x) Real loss: %.4f\tD(G(z)) Fake loss: %.4f'
                        % (epoch+1, n_epochs, batch_idx, len(dataloader),
                            loss_d.item(), loss_g.item(), loss_d_real.item(), loss_d_fake.item()))
                    writer.add_scalars("Losses", {"D real Loss": loss_d_real.item(), "D fake Loss": loss_d_fake.item(), "G Loss": loss_g.item()}, itrs )

                if batch_idx % 100 == 0:

                    writer.add_figure(
                        "fake",
                        gen_cmap( vutils.make_grid(torch.log(fake.data[:64]+1e-30) )),
                        itrs
                    )
                    
                    writer.add_figure(
                        "real", 
                        gen_cmap( vutils.make_grid(torch.log(images.data[:64]+1e-30) )), 
                        itrs
                    )

            
            itrs += 1

        loss_all_d.append([loss_d_real_epoch, loss_d_fake_epoch])
        loss_all_g.append(loss_g_epoch)
        print(f"epoch: {epoch: 4d}, loss_d_real:{np.mean(loss_d_real_epoch): .4f}, loss_d_fake: {np.mean(loss_d_fake_epoch): .4f}, loss_d: {np.mean(loss_d_epoch): .4f}, loss_g: {np.mean(loss_g_epoch): 6.4f}")
        writer.add_scalars("Epoch losses", {"D real Loss": np.mean(loss_d_real_epoch), "D fake Loss": np.mean(loss_d_fake_epoch), "G Loss": np.mean(loss_g_epoch)}, epoch+1 )
    return G, D, loss_all_d, loss_all_g



if __name__ == "__main__":

    Data_paths = glob("Data_generation/Real_data_train/slf_mat/*.mat")
    SLF_ROOT = "Data_generation/Real_data_train/"
    train_set_slf = SLFDataset(SLF_ROOT + 'slf_mat/', len(Data_paths), device="cpu")

    # Number of workers for dataloader
    workers = 8
    # Batch size during training
    batch_size = 256

    n_epochs = 500
    lr = 2e-5  
    beta1 = 0.5
    z_dimension = 256
    n_critic = 1
    torch.manual_seed(777)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(train_set_slf, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    print(f"len dataset: {len(train_set_slf)}, len dataloader: {len(dataloader)}")
    # Create the generator
    netG = Generator().to(device)
    if (device == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    netD = Discriminator().to(device)
    if (device == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netD.weight_init(mean=1.0, std=0.02)
    netG.weight_init(mean=0.0, std=0.02)

    # Print the network details
    # print(netG)
    # print(netD)

    #Train generator and discriminator
    g, d, loss_d, loss_g = train(dataloader, z_dimension, netG, netD, n_epochs, lr, n_critic, beta1, device)
    writer.close()

    #Save model
    torch.save(g.state_dict(), "Models/GAN/"+now+"_generator.pt")
    torch.save(d.state_dict(), "Models/GAN/"+now+"_discriminator.pt")
    print("Model saved successfully.")