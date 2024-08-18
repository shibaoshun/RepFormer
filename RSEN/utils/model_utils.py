import torch
import os
from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # for p in optimizer.param_groups: lr = p['lr']
    # return lr
def get_arch(opt):
    from rsen import RSENet

    model_restoration =RSENet(dim=opt.TRAINING.DIM, bias=opt.TRAINING.BIAS)

    return model_restoration

def get_test(opt):
    from rsen import RSENet
    model_restoration = RSENet(dim=opt.dim, bias=opt.bias)
    return model_restoration
def get_arch1(opt):
    from rsen import RSENet

    model_restoration =RSENet(dim=opt.TRAINING.DIM, head_dim=opt.TRAINING.HEAD_DIM)

    return model_restoration
def get_test1(opt):
    from rsen import RSENet

    model_restoration =RSENet(dim=opt.dim, head_dim=opt.head_dim)

    return model_restoration
