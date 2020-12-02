import pickle as pk
import torch
import torch.utils.data as Data
from tqdm import tqdm
from model import GCN
import numpy as np
import argparse
from collections import OrderedDict


args = argparse.ArgumentParser()
args.add_argument('--lr', type=float, default=5e-5)
args.add_argument('--embedding_dim', type=int, default=64)
args.add_argument('--hidden1', type=int, default=32)
args.add_argument('--hidden2', type=int, default=16)
args.add_argument('--batch_size', type=int, default=256)
args.add_argument('--seed', type=int, default=0)
args.add_argument('--exp_id', default='default')
args.add_argument('--droprate', type=float, default=0.5)
args.add_argument('--epoch', type=int)

opt = args.parse_args()


def create_dataloader():
    with open('./test_data.pkl', 'rb') as f:
        test_data = pk.load(f)
        test_adj = test_data['adj']
        test_element = test_data['element']
        test_hcount = test_data['hcount']
        test_charge = test_data['charge']
        test_aromatic = test_data['aromatic']
        test_mask = test_data['mask']
    name = np.load('./data/test/names_onehots.npy', allow_pickle=True).item()['names']
    test_dataset = Data.TensorDataset(test_element, test_hcount, test_charge, test_aromatic, test_adj, test_mask)
    test_sampler = Data.SequentialSampler(test_dataset)
    test_loader = Data.DataLoader(test_dataset, sampler=test_sampler, batch_size=256)

    return test_loader, name

    
if __name__ == '__main__':
    # m = GCN(64 * 3, opt.hidden1, opt.hidden2, 1, opt.droprate, opt.embedding_dim)
    m = GCN(64 * 4, opt.hidden1, opt.hidden2, 1, opt.droprate, opt.embedding_dim)
    m.to(torch.device("cuda"))
    m = torch.nn.DataParallel(m)
    saved_state_path = f'./exp/{opt.exp_id}/{opt.exp_id}_best_epoch{opt.epoch}.pth'
    state_dict = torch.load(saved_state_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = 'module.' + k
        new_state_dict[k] = v
    m.load_state_dict(new_state_dict)

    test_loader, name = create_dataloader()
    all_preds = []

    with torch.no_grad():
        m.eval()
        for element, hcount, charge, aromatic, adj, mask in tqdm(test_loader):
            element = element.to(torch.device('cuda'))
            hcount = hcount.to(torch.device('cuda'))
            charge = charge.to(torch.device('cuda'))
            aromatic = aromatic.to(torch.device('cuda'))
            adj = adj.float().to(torch.device('cuda'))
            mask = mask.float().to(torch.device('cuda'))
            output = m(element, hcount, charge, adj, aromatic, mask)
            # output = m(element, hcount, charge, adj, aromatic, None)
            # output = m(element, hcount, charge, adj, None, None)
            output = output[:, 1]
            for item in output:
                all_preds.append(item.item())
    
    with open(f'./exp/{opt.exp_id}/output_518030910173_{opt.epoch}.txt', 'w') as f:
        f.write('Chemical,Label\n')
        for i in range(len(all_preds)):
            line = f'{name[i]},{all_preds[i]}\n'
            f.write(line)

    with open('output_518030910173.txt', 'w') as f:
        f.write('Chemical,Label\n')
        for i in range(len(all_preds)):
            line = f'{name[i]},{all_preds[i]}\n'
            f.write(line)