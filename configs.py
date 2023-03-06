import torch
import numpy as np


def create_mask_matrix(f, fiber, H=51, W=51, K=64):
    #Create mask for sampling measurements
    # sampling frequency (fraction of measurements )
    # Mask

    if fiber:
        # print("Fiber sampling.")
        Om = ( torch.bernoulli(torch.ones((1, H, W))*f) ).repeat(K,1,1,1)
    else:
        # print("Space sampling.")
        Om = ( torch.ones((K,1,H,W))*f ) 
        Om = torch.bernoulli(Om)

    return Om

class SelectParameters():
    
    def __init__(self, bits):
        self.bits = bits
        
        ###### Quantization parameters #######################
        self.OFFSET = 1e-6

        self.QUANTIZATION_BINS = torch.tensor( [-23.025850296020508, -11.350225067138672, -10.472214698791504, -9.010324974060059, -8.931082344055176,
                         -8.040789890289307, -7.61128044128418, -5.762726783752441, -1.2379993200302124])

        self.std_probit = 1.7

        ################ Learning rates #######################
        self.alpha = 0.003 
        self.beta = 0.006 

        ################# Regularization Parameters ################
        self.lambdda =  1e-3
        








