import torchvision.models as tm
import pandas as pd
import pickle as pk
import torch
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as Data
import random
import logging
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from collections import OrderedDict
import argparse
# import sys
# sys.path.append('../')
# from .resnet_model import resnet10
import resnet_model


def create_dataset():
    with open('../../data/train/train_data.pkl', 'rb') as f:
        train_data = pk.load(f)
        train_label = train_data['label']
        train_data = train_data['data']
    with open('../../data/validation/validation_data.pkl', 'rb') as f:
        val_data = pk.load(f)
        val_label = val_data['label']
        val_data = val_data['data']
    train_dataset = Data.TensorDataset(train_data, train_label)
    val_dataset = Data.TensorDataset(val_data, val_label)
    true_num = sum(train_label).item()
    false_num = len(train_label) - true_num
    weight = []
    for item in train_label:
        if item.item() == 0:
            weight.append(1.0 / false_num)
        elif item.item() == 1:
            weight.append(1.0 / true_num)
        else:
            print('fuck')
            print(item)
            quit()
    train_sampler = Data.WeightedRandomSampler(weight, num_samples=len(weight), replacement=True)
    # train_sampler = Data.WeightedRandomSampler(weight, num_samples=len(weight) * 10, replacement=True)
    train_loader = Data.DataLoader(train_dataset, sampler=train_sampler, batch_size=256)
    val_sampler = Data.SequentialSampler(val_dataset)
    val_loader = Data.DataLoader(val_dataset, sampler=val_sampler, batch_size=272)

    return train_loader, val_loader


def setup_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

# class ResnetModel(nn.Module):
#     def __init__(self) -> None:
#         '''
#         input size: B x 3 x 73 x 398
#         '''
#         super(ResnetModel, self).__init__()
#         self.resnet = tm.resnet18()
#         # self.resnet = tm.resnet50()
#         self.fc1 = nn.Linear(1000, 250)
#         self.drop1 = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(250,2)
#         self.drop2 = nn.Dropout(p=0.5)
#         self.softmax = nn.Softmax(dim=1)

    
#     def forward(self, x):
#         x = self.resnet(x)
#         x = self.fc1(x)
#         x = self.drop1(x)
#         x = self.fc2(x)
#         x = self.drop2(x)

#         x = self.softmax(x)

#         return x


def train(m, criterion, optimizer, train_loader):
    m.train()
    train_loader = tqdm(train_loader, dynamic_ncols=True)
    for x, y in train_loader:
        x = x.float().to(torch.device("cuda"))
        y = y.float().to(torch.device("cuda"))
        # print(x.dtype)
        output = m(x)
        output = output[:, 1]
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        auc_score = roc_auc_score(y.clone().detach().cpu(), output.clone().detach().cpu())
        train_loader.set_description('loss: {loss:.8f} | auc score: {auc_score:.4f}'
            .format(loss=loss, auc_score=auc_score))
        # print(loss.item())


def evaluate(m, val_loader):
    m.eval()
    for x, y in val_loader:
        x = x.float().to(torch.device("cuda"))
        y = y.float().to(torch.device("cuda"))
        output = m(x)
        output = output[:, 1]
        auc_score = roc_auc_score(y.cpu(), output.cpu())

    return auc_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Drug Molecular Toxicity Prediction Training')
    parser.add_argument('--exp_id', default='default', type=str, help='Experiment ID')
    parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--epoch_num', default=3, type=int, help='number of training epoch')
    opt = parser.parse_args()
    setup_seed(opt.seed)
    if not os.path.exists(f'./exp/{opt.exp_id}'):
        os.makedirs(f'./exp/{opt.exp_id}')
    train_loader, val_loader = create_dataset()
    m = resnet_model.resnet50()
    # m = resnet_model.resnet34()
    # m = resnet_model.resnet18()
    # m = resnet_model.resnet10()
    m.to(torch.device("cuda"))
    m = torch.nn.DataParallel(m)

    filehandler = logging.FileHandler(f'./exp/{opt.exp_id}/training.log')
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    
    epoch_limit = 30
    lr = opt.lr

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    milestones=[10, 20]
    best_auc_score = 0

    for epoch in range(epoch_limit):
        if epoch in milestones:
            print('loading previous best model...')
            state_dict = torch.load(f'./exp/{opt.exp_id}/{opt.exp_id}_best.pth')
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                k = 'module.' + k
                new_state_dict[k] = v
            m.load_state_dict(new_state_dict)
            with torch.no_grad():
                auc_score = evaluate(m, val_loader)
            print(f'epoch {epoch}\'s auc score: {auc_score:.4f}')
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'############# Starting Epoch {epoch} | LR: {current_lr} #############')
        train(m, criterion, optimizer, train_loader)
        with torch.no_grad():
            auc_score = evaluate(m, val_loader)
        if auc_score >= best_auc_score:
            best_auc_score = auc_score
            print('Saving best model...')
            torch.save(m.module.state_dict(), f'./exp/{opt.exp_id}/{opt.exp_id}_best.pth')
        lr_scheduler.step()
        print(f'epoch {epoch}\'s auc score: {auc_score:.4f} | best auc score: {best_auc_score:.4f}')

    torch.save(m.module.state_dict(), f'./exp/{opt.exp_id}/{opt.exp_id}_final.pth')        
