import torch
from quantization_model_log import prob_probit


class CostFunction():
    
    def __init__(self, sample_matrix, reg, method, bins, std_val, device):
        
        self.method = method
        self.Wx = sample_matrix
        
        self.lambda_c = reg  # For Z
        self.lambda_z1 = reg # For A or Z
        self.lambda_z2 = reg # For B
        
        #For parameters for calculating loss
        self.bin_boundaries = bins
        self.device = device
        self.std_probit = std_val
        
        
    def calculate_cost(self, Y, T_hat, C, Z):
        #Negative log-likelihood
        cost_specific = - torch.sum(self.Wx * torch.log(prob_probit(Y, T_hat, self.bin_boundaries, self.std_probit)))
        #Regularizations
        if self.method == "btd":
            cost = cost_specific
        else:
            cost = cost_specific
            
            
        return cost