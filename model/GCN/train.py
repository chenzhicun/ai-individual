import pickle as pk
import torch
import torch.utils.data as Data
import random
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import argparse
import os
import logging
from model import GCN, GCN_4layer, GCN_4layer_relu, GCN_5layer, GCN_8layer


args = argparse.ArgumentParser()
args.add_argument('--epochs', type=int, default=30)
args.add_argument('--lr', type=float, default=5e-5)
args.add_argument('--embedding_dim', type=int, default=64)
args.add_argument('--hidden1', type=int, default=32)
args.add_argument('--hidden2', type=int, default=16)
args.add_argument('--hidden3', type=int, default=8)
args.add_argument('--hidden4', type=int, default=4)
args.add_argument('--batch_size', type=int, default=256)
args.add_argument('--seed', type=int, default=0)
args.add_argument('--exp_id', default='default')
args.add_argument('--droprate', type=float, default=0.5)

opt = args.parse_args()

if not os.path.exists(f'./exp/{opt.exp_id}'):
        os.makedirs(f'./exp/{opt.exp_id}')

filehandler = logging.FileHandler(f'./exp/{opt.exp_id}/training.log', mode='w')
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)


def setup_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_dataloader():
    with open('./train_data.pkl', 'rb') as f:
        train_data = pk.load(f)
        train_label = train_data['labels']
        train_adj = train_data['adj']
        train_element = train_data['element']
        train_hcount = train_data['hcount']
        train_charge = train_data['charge']
        train_aromatic = train_data['aromatic']
        train_mask = train_data['mask']
    with open('./validation_data.pkl', 'rb') as f:
        val_data = pk.load(f)
        val_label = val_data['labels']
        val_adj = val_data['adj']
        val_element = val_data['element']
        val_hcount = val_data['hcount']
        val_charge = val_data['charge']
        val_aromatic = val_data['aromatic']
        val_mask = val_data['mask']

    train_dataset = Data.TensorDataset(train_element, train_hcount, train_charge, train_aromatic, train_adj, train_label, train_mask)
    val_dataset = Data.TensorDataset(val_element, val_hcount, val_charge, val_aromatic, val_adj, val_label, val_mask)

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
    train_loader = Data.DataLoader(train_dataset, sampler=train_sampler, batch_size=opt.batch_size)
    val_sampler = Data.SequentialSampler(val_dataset)
    val_loader = Data.DataLoader(val_dataset, sampler=val_sampler, batch_size=272)

    return train_loader, val_loader


def train(m ,criterion, optimizer, train_loader):
    m.train()
    train_loader = tqdm(train_loader, dynamic_ncols=True)
    for element, hcount, charge, aromatic, adj, label, mask in train_loader:
        element = element.to(torch.device('cuda'))
        hcount = hcount.to(torch.device('cuda'))
        charge = charge.to(torch.device('cuda'))
        aromatic = aromatic.to(torch.device('cuda'))
        adj = adj.float().to(torch.device('cuda'))
        label = label.float().to(torch.device('cuda'))
        mask = mask.float().to(torch.device('cuda'))

        output = m(element, hcount, charge, adj, aromatic, mask)
        # output = m(element, hcount, charge, adj, aromatic, None)
        # output = m(element, hcount, charge, adj, None, None)
        output = output[:, 1]
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        auc_score = roc_auc_score(label.clone().detach().cpu(), output.clone().detach().cpu())
        train_loader.set_description('loss: {loss:.8f} | auc score: {auc_score:.4f}'
            .format(loss=loss, auc_score=auc_score))


def evaluate(m, val_loader):
    m.eval()
    for element, hcount, charge, aromatic, adj, label, mask in val_loader:
        element = element.to(torch.device('cuda'))
        hcount = hcount.to(torch.device('cuda'))
        charge = charge.to(torch.device('cuda'))
        aromatic = aromatic.to(torch.device('cuda'))
        adj = adj.float().to(torch.device('cuda'))
        label = label.float().to(torch.device('cuda'))
        mask = mask.float().to(torch.device('cuda'))

        output = m(element, hcount, charge, adj, aromatic, mask)
        # output = m(element, hcount, charge, adj, aromatic, None)
        # output = m(element, hcount, charge, adj, None, None)
        output = output[:, 1]
        auc_score = roc_auc_score(label.cpu(), output.cpu())

    return auc_score


if __name__ == '__main__':
    setup_seed(opt.seed)
    train_loader, val_loader = create_dataloader()
    # m = GCN(64 * 3, opt.hidden1, opt.hidden2, 1, opt.droprate, opt.embedding_dim)
    # m = GCN(64 * 4, opt.hidden1, opt.hidden2, 1, opt.droprate, opt.embedding_dim)
    # m = GCN_4layer(64 * 4, opt.hidden1, opt.hidden2, opt.hidden3, 1, opt.droprate, opt.embedding_dim)
    # m = GCN_4layer_relu(64 * 4, opt.hidden1, opt.hidden2, opt.hidden3, 1, opt.droprate, opt.embedding_dim)
    # m = GCN_5layer(64 * 4, opt.hidden1, opt.hidden2, opt.hidden3, opt.hidden4, 1, opt.droprate, opt.embedding_dim)
    m = GCN_8layer()
    m.to(torch.device("cuda"))
    m = torch.nn.DataParallel(m)

    record_best = {}
    epoch_limit = opt.epochs
    lr = opt.lr
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    best_auc_score = 0
        
    for epoch in range(epoch_limit):
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(f'############# Starting Epoch {epoch} | LR: {current_lr} #############')
        train(m, criterion, optimizer, train_loader)
        with torch.no_grad():
            auc_score = evaluate(m, val_loader)
        if auc_score >= best_auc_score:
            best_auc_score = auc_score
            logger.info('Saving best model...')
            torch.save(m.module.state_dict(), f'./exp/{opt.exp_id}/{opt.exp_id}_best_epoch{epoch}.pth')
            record_best[epoch] = auc_score
        logger.info(f'epoch {epoch}\'s auc score: {auc_score:.4f} | best auc score: {best_auc_score:.4f}')

    torch.save(m.module.state_dict(), f'./exp/{opt.exp_id}/{opt.exp_id}_final.pth')
    with open(f'./exp/{opt.exp_id}/record_best.txt', 'w') as f:
        f.write('epoch, auc score\n')
        for key, item in record_best.items():
            f.write(f'{key}, {item}\n')     