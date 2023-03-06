import torch
from quantization_model_log import prob_probit


class CostFunction():
    
    def __init__(self, sample_matrix, reg, method, bins, std_val, device, use_reg = True):
        
        self.method = method
        self.use_reg = use_reg
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
        if self.use_reg:
            if self.method == "btd":
                R = Z.shape[0]//2
                cost = cost_specific + self.lambda_z1 * torch.norm(Z[:R,:,:], p='fro') + self.lambda_z2 * torch.norm(Z[R:,:,:], p='fro') \
                        + self.lambda_c * torch.norm(C, p='fro')
            else: #For DGM
                cost = cost_specific + self.lambda_z1 * torch.norm(Z, p='fro') + self.lambda_c * torch.norm(C, p='fro')      
            
        return cost