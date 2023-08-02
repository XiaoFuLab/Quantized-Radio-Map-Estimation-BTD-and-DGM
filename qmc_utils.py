import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.io as sio
from gan import Generator256



def load_data(DATA_PATH, device):
    data = sio.loadmat(DATA_PATH)
    
    S = torch.from_numpy(data['S']).type(torch.float32)
    T = torch.from_numpy(data['T']).type(torch.float32)
    C = torch.from_numpy(data['C']).type(torch.float32)
    
    S_true = torch.from_numpy(data['S_true']).type(torch.float32)
    C_true = torch.from_numpy(data['C_true']).type(torch.float32)
    T_true = torch.from_numpy(data['T_true']).type(torch.float32)
    
    # Permutation for compatibility with matlab generated arrays
    T = T.permute(2,0,1).to(device)
    T_true = T_true.permute(2,0,1).to(device)
    try:
        S = S.permute(2,0,1).to(device)
    except:
        S = S.unsqueeze(dim=0)
    
    try:
        S_true = S_true.permute(2,0,1).to(device)
    except:
        S_true = S_true.unsqueeze(dim=0)
    
    
    C = C.permute(1,0).to(device)
    C_true = C_true.permute(1,0).to(device)
    
    return S, C, T, S_true, C_true, T_true


    
    
def load_generator(GAN_PATH, device):
    generator = Generator256()
    
    checkpoint = torch.load(GAN_PATH, map_location=torch.device('cpu'))
        
    generator.load_state_dict(checkpoint['g_model_state_dict'])
    generator.eval()
    return generator.to(device)

def load_all( datapath, z_dimension, device, visualize_data = True, k=25):

    S, C, T, S_true, C_true, T_true = load_data(datapath, device)

    # C_true[0:,50:] = 0
    # T_true = get_tensor(S_true.unsqueeze(dim=1), C_true, device)

    if visualize_data:
        print(S.shape[0], "emitters.")
        fig, ax = plt.subplots(1, C.shape[0]+1, figsize=(5*C.shape[0], 3))
        
        for i in range(C.shape[0]):
            ax[i].plot(C_true[i,:].cpu().detach().numpy())
            
        ax[i+1].imshow(torch.log(T_true[k,...]).cpu() )

    # parameters
    R, I, J = S.shape # R is the number of emitters, I,J is the size of the SLF
    K = C.shape[-1] # K is the number of frequency bands

    # Initialize the latent vectors for each emitter
    Z_init = torch.randn((R, z_dimension), dtype=torch.float32).to(device) 
    Z = torch.zeros((R, z_dimension), dtype=torch.float32).to(device) 

    # zero start 
    S_init = torch.zeros( (R, 1, I, J) ).to(device) 
    C_init = torch.zeros(C.shape).to(device)

    T = T.unsqueeze(dim=1)

    return S_init, C_init, T, S_true, C_true, T_true, Z_init, Z