import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
        
    def forward(self, input):
        return torch.reshape(input, (input.size(0),*self.target_shape))


class Generator256(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 16
        self.main = nn.Sequential(
            
            UnFlatten((ndf*16, 1, 1)),
            # state size. (ndf*8) x 1 x 1
            nn.ConvTranspose2d(ndf*16, ndf*8, 3, 1, 0),
            nn.BatchNorm2d(ndf*8),
            nn.ReLU(True),
            
            # state size. (ndf*8) x 3 x 3
            nn.ConvTranspose2d(ndf*8, ndf*4, 4, 2, 1),
            nn.BatchNorm2d(ndf*4),
            nn.ReLU(True),
            
            # state size. (ndf*4) x 6 x 6   
            nn.ConvTranspose2d(ndf*4, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.ReLU(True),

            # state size (ndf*2) x 12 x 12
            nn.ConvTranspose2d(ndf*2, ndf*1, 4, 2, 0),
            nn.BatchNorm2d(ndf*1),
            nn.ReLU(True),

            # state size (ndf) x 26, 26
            nn.ConvTranspose2d(ndf*1,2, 4, 2, 0),
            nn.BatchNorm2d(2),
            nn.ReLU(True),

            # state size 2 x 54 x 54
            nn.Conv2d(2,1, 4, 1, 0),
            nn.Sigmoid()
            # output 1 x 51 x 51              
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        ndf = 16
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is 1 x 51 x 51
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 25 x 25
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 12 x 12
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 6 x 6
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 3 x 3
            nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)