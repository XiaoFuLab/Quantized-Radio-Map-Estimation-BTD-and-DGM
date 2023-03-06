import torch
    
class TransformLog():
    
    def __init__(self, offset, scale=1.0):
        self.offset = offset
        self.scale = scale
        
    def trans(self, X):
        return torch.log(self.scale*X + self.offset)
    
    def inverse_trans(self, X_inv):
        return (torch.exp(X) - self.offset)/self.scale