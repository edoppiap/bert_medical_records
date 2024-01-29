from torch.optim import AdamW

def get_optimizer(parameters, lr = 5e-5):

    optim = AdamW(parameters, lr=lr)
    
    return optim