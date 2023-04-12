import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import neptune.new as neptune
import numpy as np

from src.dataset import build_transforms, make_train_loaders
from src.model import LandmarkNet
from src.utils import get_optim, AverageMeter, accuracy, save_checkpoint
from src.MCS_validation import product_matching_validation
from config import params

run = neptune.init_run(
    project="vmudryi/Product-matching",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0Njk1ZDE5ZS02ODhhLTRhMmYtYjRhNC0wZTlhNjBkYWYzNTUifQ==",
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

run['params'] = params
run['params/img_size/height'] = params['image_size'][0]
run['params/img_size/width'] = params['image_size'][1]

np.random.seed(params['seed'])
torch.manual_seed(params['seed'])

train_transform, eval_transform = build_transforms(
    scale_limit=params['scale_limit'],
    shear_limit=params['shear_limit'],
    brightness_limit=params['brightness_limit'],
    contrast_limit=params['contrast_limit'],
    # mean=(0.5, 0.5, 0.5), # for beit
    # std=(0.5, 0.5, 0.5)
)

data_loaders = make_train_loaders(params=params,
                                  data_root='train',
                                  train_transform=train_transform,
                                  eval_transform=eval_transform,
                                  scale='S2',
                                  test_size=params['test_size'],
                                  num_workers=1
                                  )

model = LandmarkNet(n_classes=params['class_topk'],
                    model_name=params['model_name'],
                    pooling=params['pooling'],
                    loss_module=params['loss'],
                    s=params['s'],
                    margin=params['margin'],
                    theta_zero=params['theta_zero'],
                    use_fc=params['use_fc'],
                    use_prelu=params['use_prelu'],
                    fc_dim=params['fc_dim']
                    ).to(device)

optimizer = get_optim(params, model)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                 T_max=params['epochs'] * len(data_loaders['train']),
                                                 eta_min=params['scheduler_eta_min'])

start_epoch = 0
best_mape = 0

for epoch in range(start_epoch, params['epochs']):
    print(f'Epoch {epoch}/{params["epochs"]} | lr: {optimizer.param_groups[0]["lr"]}')
    model.train(True)
    losses = AverageMeter()
    prec1 = AverageMeter()

    for i, (_, x, y) in tqdm(enumerate(data_loaders['train']),
                             total=len(data_loaders['train']),
                             miniters=None, ncols=55):
        x = x.to(device)
        y = y.to(device)

        outputs = model(x, y)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc = accuracy(outputs, y)
        losses.update(loss.item(), x.size(0))
        prec1.update(acc, x.size(0))

        if i % 100 == 99:
            print(f'{epoch + i / len(data_loaders["train"]):.2f}epoch | acc: {prec1.avg}')
            run["train/acc"].append(round(prec1.avg, 10))
            run["train/loss"].append(round(losses.avg, 10))

    train_loss = losses.avg
    train_acc = prec1.avg

    print('Loss', {'train': train_loss}, epoch)
    print('Acc', {'train': train_acc}, epoch)
    print('LR', optimizer.param_groups[0]['lr'], epoch)
    run['train/lr'].append(round(optimizer.param_groups[0]['lr'], 14))

    target_mape = product_matching_validation(model, embedding_size=params['fc_dim'])
    print('Map MSC = ', target_mape)

    run['eval/Map'].append(round(target_mape, 10))

    eval_loss = []
    for i, (_, x, y) in tqdm(enumerate(data_loaders['val']), total=len(data_loaders['val']), miniters=None, ncols=55):
        model = model.eval()

        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            outputs = model(x, y)
            loss = criterion(outputs, y)
        eval_loss.append(loss.item())

    run['eval/loss'].append(round(np.mean(eval_loss), 10))

    if best_mape < target_mape:
        best_mape = target_mape
        save_checkpoint(path="best_model.pth",
                        model=model,
                        epoch=epoch
                        )

run.stop()
