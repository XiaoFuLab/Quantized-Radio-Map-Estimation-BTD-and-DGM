import os
import torch
import torch.nn as nn
import scipy.io as sio
from torch.utils.data import Dataset

class SLFDataset(Dataset):
    """SLF loader"""

    def __init__(self, root_dir, total_data, device):
        """
        Args:
            total_data: Number of data points
        """
        self.root_dir = root_dir
        self.num_examples = total_data
        self.device = device
        
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        str_idx = str(idx+1)
        data_name = ['0', '0', '0', '0', '0', '0', '0']
        data_name[-len(str_idx):] =  list(str_idx)
        data_name = "".join(data_name) + ".mat"

        filename = os.path.join(self.root_dir, data_name)
        sample = torch.Tensor(sio.loadmat(filename)["Sc"] ).to(self.device)
        
        return sample.unsqueeze(dim= 0)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
        
    def forward(self, input):
        return torch.reshape(input, (input.size(0),*self.target_shape))

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        ndf = 16
        self.main = nn.Sequential(
            #input is batchsize x 1 x 256
            UnFlatten((ndf*16, 1, 1)),
            #After unflatten batchsize x 256 x 1 x 1

            nn.ConvTranspose2d(ndf*16, ndf*8, 3, 1, 0),
            nn.BatchNorm2d(ndf*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ndf*8, ndf*4, 3, 2, 1 ),
            nn.BatchNorm2d(ndf*4),
            nn.ReLU(True),
              
            nn.ConvTranspose2d(ndf*4, ndf*2, 3, (1, 2), (0, 1) ),
            nn.BatchNorm2d(ndf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ndf*2, ndf, 3, 2, 1),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ndf, ndf//2, 2, (1, 2), 0 ),
            nn.BatchNorm2d(ndf//2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ndf//2, 4, 2, 1, 0),
            nn.BatchNorm2d(4),
            nn.ReLU(True),

            nn.Conv2d(4, 1, 2, 1, 0),
            nn.Sigmoid()
            # output 1 x 14 x 34             
        )

    def forward(self, input):
        return self.main(input)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)



class Discriminator(nn.Module):
    def __init__(self):
        ndf = 16
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input size: batchsize x 1 x 14 x 34
            nn.Conv2d(1, ndf, 2, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 1, (1,0), bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, ndf * 16, 3, 2, (1, 0), bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
     
            nn.Conv2d(ndf * 16, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
            # output size: batchsize x 1 x 1 x 1
        )

    def forward(self, input):
        x = torch.log(input + 1e-30)
        return self.main(x)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)




# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         ndf = 16
#         self.main = nn.Sequential(
            
#             UnFlatten((ndf*16, 1, 1)),

#             nn.ConvTranspose2d(ndf*16, ndf*8, 3, 1, 0),
#             nn.BatchNorm2d(ndf*8),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(ndf*8, ndf*4, 3, 2, 1 ),
#             nn.BatchNorm2d(ndf*4),
#             nn.ReLU(True),
              
#             nn.ConvTranspose2d(ndf*4, ndf*2, 3, (1, 2), (0, 1) ),
#             nn.BatchNorm2d(ndf*2),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(ndf*2, ndf, 3, 2, 1),
#             nn.BatchNorm2d(ndf),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(ndf, ndf//2, 2, (1, 2), 0 ),
#             nn.BatchNorm2d(ndf//2),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(ndf//2, 4, 2, 1, 0),
#             nn.BatchNorm2d(4),
#             nn.ReLU(True),

#             nn.Conv2d(4, 1, 2, 1, 0),
#             nn.Sigmoid()
#             # output 1 x 14 x 34             
#         )


# class Discriminator(nn.Module):
#     def __init__(self):
#         ndf = 16
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             # input is 1 x 14 x 34
#             nn.Conv2d(1, ndf, 2, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 25 x 25

#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),

#             # # state size. (ndf*2) x 12 x 12
#             nn.Conv2d(ndf * 2, ndf * 4, 2, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),

#             # # state size. (ndf*4) x 6 x 6
#             nn.Conv2d(ndf * 4, ndf * 8, 2, 2, (1, 0), bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),

#             # state size. (ndf*8) x 3 x 3
#             nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
#             # state size. 1 x 1 x 1
#             nn.Sigmoid()
#         )