import torch
import numpy as np


def quantize(X, noise_std, bin_boundaries, device):
    """
    Returns quantized observation of the ground truth with respect to the bin boundaries following the model: Y = Q(X + E), E ~ N(0, noise_std)
    Values of Y are the bin indices: e.g., Y[i,j] = 2 if X+E lies in between bin_boundaries[2] and bin_boundaries[3]
    """
    noise_X = X + torch.randn(X.shape).to(device)*noise_std
    Y = torch.zeros(X.shape).to(device)
    bin_boundaries = bin_boundaries.clone()
    bin_boundaries[-1] = np.inf
    for i in range(1, len(bin_boundaries)-1):
        indices = torch.logical_and((bin_boundaries[i] < noise_X), (noise_X <= bin_boundaries[i+1]))
        Y[indices] = i
    return Y.long()


def quantize_value(X, bin_boundaries, device):
    
    bin_boundaries = bin_boundaries.clone()
    bin_boundaries[-1] = np.inf
    
    Y = torch.ones(X.shape).to(device)
    Y = Y * bin_boundaries[0]
    
    for i in range(1, len(bin_boundaries)-1):
        indices = torch.logical_and((bin_boundaries[i] < X), (X <= bin_boundaries[i+1]))
        Y[indices] = (bin_boundaries[i] + bin_boundaries[i-1])/2
    return Y

def prob_probit(Y, X_hat, bin_boundaries, noise_std):
    """
    Calculates the likelihood of the observation under the current estimate of the signal
    p(Y|X) = Phi(U-X) - phi(W-X), where U is the lower boundary tensor and W the upper boundary tensor

    @params: 
        X_hat : Current estimate of the ground truth
        Y : Observation tensor with the boundary indices
    """
    bin_boundaries = bin_boundaries.clone()
    W = bin_boundaries[Y]
    U = bin_boundaries[Y+1]

    # print(U-X_hat)
    P = F_probit(U-X_hat, noise_std) - F_probit(W-X_hat, noise_std)

    return P+1e-15
    

def F_probit(y, std):
    """
    Evaluates the probit function value for the given input
    """
    return (1/2)*(1 + torch.erf(y/(std*1.414213)))


############################Tensor Products#############################

def outer(mat : torch.Tensor, vec : torch.Tensor, device):
    """
    Compute outer product given a matrix and a vector
    """
    prod = torch.zeros(( *vec.shape,*mat.shape), dtype=torch.float32).to(device)
    for i in range(len(vec)):
        prod[i,:,:] = mat*vec[i]
    return prod

def get_tensor(S: torch.Tensor, C: torch.Tensor, device):
    """
    Returns  sum_i (S[i] o C[i])
    """
    prod = 0
    for i in range(C.shape[0]):
        prod += outer(S[i,0,:,:], C[i,:], device)
    return prod

################## Metric #######################################

def NMSE_LOG(T: torch.Tensor, T_target: torch.Tensor, offset=1e-6):
    """
    Normalized mean squared error after taking log
    """
    T_log = torch.log(T+offset).clone().detach()
    T_target_log = torch.log(T_target+offset).clone().detach()

    return torch.norm(T_log-T_target_log, 'fro')/torch.norm(T_target_log, 'fro')
