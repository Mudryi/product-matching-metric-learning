import torch.nn as nn
import torch.optim as optim
import torch


def get_optim(params, target):
    assert isinstance(target, nn.Module) or isinstance(target, dict)

    if isinstance(target, nn.Module):
        target = target.parameters()

    if params['optimizer'] == 'sgd':
        optimizer = optim.SGD(target, params['lr'], weight_decay=params['wd'])
    elif params['optimizer'] == 'momentum':
        optimizer = optim.SGD(target, params['lr'], momentum=0.9, weight_decay=params['wd'])
    elif params['optimizer'] == 'nesterov':
        optimizer = optim.SGD(target, params['lr'], momentum=0.9,
                              weight_decay=params['wd'], nesterov=True)
    elif params['optimizer'] == 'adam':
        optimizer = optim.Adam(target, params['lr'], weight_decay=params['wd'])
    elif params['optimizer'] == 'amsgrad':
        optimizer = optim.Adam(target, params['lr'], weight_decay=params['wd'], amsgrad=True)
    elif params['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(target, params['lr'], weight_decay=params['wd'])
    else:
        raise ValueError

    return optimizer


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = preds.eq(labels)
    acc = correct.float().mean().item()
    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = None
        self.avg = None
        self.val = None
        self.sum = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(path, model, epoch, save_arch=False):
    attributes = {
        'epoch': epoch,
        'state_dict': model.state_dict()
    }

    if save_arch:
        attributes['arch'] = model

    try:
        torch.save(attributes, path)
    except TypeError:
        if 'arch' in attributes:
            print('Model architecture will be ignored because the architecture includes non-pickable objects.')
            del attributes['arch']
            torch.save(attributes, path)
