from torch.optim import AdamW

def get_optimizer(parameters, lr = 5e-5, eps = 5e-5):

    optim = AdamW(parameters, lr=lr, eps=eps)
    
    return optim