import networkx
import pysmiles
import numpy as np
import torch
import os
from torch import exp
import torch.nn.functional as F
import scipy.sparse as sp
import pandas as pd
import pickle as pk


element_map_table = {'Hg': 1, 'K': 2, 'Au': 3, 'Mn': 4, 'Zn': 5, 'In': 6, 'B': 7, 'Ba': 8, 'Na': 9, 'H': 10,
                     'N': 11, 'Bi': 12, 'Dy': 13, 'Li': 14, 'S': 15, 'P': 16, 'Ni': 17, 'O': 18, 'Cl': 19, 'Pb': 20, 
                     'Zr': 21, 'Ru': 22, 'Eu': 23, 'Si': 24, 'Tl': 25, 'Cr': 26, 'Cu': 27, 'Yb': 28, 'Mo': 29, 'Se': 30, 
                     'Mg': 31, 'As': 32, 'Ti': 33, 'V': 34, 'C': 35, 'Al': 36, 'Be': 37, 'Ag': 38, 'F': 39, 'Ge': 40, 
                     'Nd': 41, 'Sb': 42, 'Fe': 43, 'Pt': 44, 'Pd': 45, 'Co': 46, 'Br': 47, 'Cd': 48, 'Ca': 49, 'Sr': 50, 
                     'I': 51, 'Sc': 52, 'Sn': 53}
charge_map_table = {0: 1, 1: 2, 2: 3, 3: 4, -2: 5, -1: 6}
hcount_map_table = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
aromatic_map_table = {False: 1, True: 2}


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def generate_data(mode, write=True, explicit_h=False):
    if explicit_h:
        max_length = 227
    else:
        max_length = 132
    
    if mode == 'train':
        root_path = './data/train'
    if mode == 'validation':
        root_path = './data/validation'
    if mode == 'test':
        root_path = './data/test'
    
    all_element_onehot = []
    if not explicit_h:
        all_hcount_onehot = []
    all_charge_onehot = []
    all_aromatic_onehot = []
    all_adj = []
    all_mask = []
    
    with open(os.path.join(root_path, 'names_smiles.txt')) as f:
        items = f.read().split('\n')[1:]
        for item in items:
            if item == '':
                continue
            name = item.split(',')[0]
            smile = item.split(',')[1]
            mol = pysmiles.read_smiles(smile, explicit_hydrogen=explicit_h)
            element_tmp = mol.nodes(data='element')
            if not explicit_h:
                hcount_tmp = mol.nodes(data='hcount')
            charge_tmp = mol.nodes(data='charge')
            aromatic_tmp = mol.nodes(data='aromatic')

            element_list = []
            if not explicit_h:
                hcount_list = []
            charge_list = []
            aromatic_list = []

            for element in element_tmp:
                atom = element_map_table[element[1]]
                element_list.append(atom)
            if not explicit_h:
                for hcount in hcount_tmp:
                    hcount_list.append(hcount_map_table[hcount[1]])
            for charge in charge_tmp:
                charge_list.append(charge_map_table[charge[1]])
            for aromatic in aromatic_tmp:
                aromatic_list.append(aromatic_map_table[aromatic[1]])
            
            element_list = np.pad(element_list, (0, max_length - len(element_list)))
            if not explicit_h:
                hcount_list = np.pad(hcount_list, (0, max_length - len(hcount_list)))
            charge_list = np.pad(charge_list, (0, max_length - len(charge_list)))
            aromatic_list = np.pad(aromatic_list, (0, max_length - len(aromatic_list)))

            mask = element_list != 0
            mask = torch.from_numpy(mask).int().unsqueeze(0)
            all_mask.append(mask)

            adj = networkx.to_numpy_array(mol, weight='order')
            adj = adj + adj.T.dot(adj.T > adj) - adj.dot(adj.T > adj)
            adj = normalize(adj + sp.eye(adj.shape[0]))

            adj = np.pad(adj, (0, max_length - len(adj)))

            all_adj.append(adj)
            # element_onehot = F.one_hot(torch.Tensor(element_list).long(), 54).unsqueeze(0)
            all_element_onehot.append(torch.LongTensor(element_list).unsqueeze(0))
            # hcount_onehot = F.one_hot(torch.Tensor(hcount_list).long(), 6).unsqueeze(0)
            if not explicit_h:
                all_hcount_onehot.append(torch.LongTensor(hcount_list).unsqueeze(0))
            # charge_onehot = F.one_hot(torch.Tensor(charge_list).long(), 7).unsqueeze(0)
            all_charge_onehot.append(torch.LongTensor(charge_list).unsqueeze(0))
            # aromatic_onehot = F.one_hot(torch.Tensor(aromatic_list).long(), 3).unsqueeze(0)
            all_aromatic_onehot.append(torch.LongTensor(aromatic_list).unsqueeze(0))

        all_adj = torch.Tensor(all_adj)
        all_element_onehot = torch.cat(all_element_onehot, dim=0)
        if not explicit_h:
            all_hcount_onehot = torch.cat(all_hcount_onehot, dim=0)
        all_charge_onehot = torch.cat(all_charge_onehot, dim=0)
        all_aromatic_onehot = torch.cat(all_aromatic_onehot, dim=0)
        all_mask = torch.cat(all_mask, dim=0)
    
    if mode != 'test':
        label = pd.read_csv(os.path.join(root_path, 'names_labels.txt'), sep=',')
        label = label['Label'].values
        label = torch.from_numpy(label)

        assert len(label) == len(all_adj)
        if write == True:
            if explicit_h:
                with open(f'{mode}_data_explicit_h.pkl', 'wb') as f:
                    pk.dump({
                        'adj': all_adj,
                        'element': all_element_onehot,
                        'charge': all_charge_onehot,
                        'aromatic': all_aromatic_onehot,
                        'mask': all_mask,
                        'labels': label
                    },f,pk.HIGHEST_PROTOCOL)
            else:
                with open(f'{mode}_data.pkl', 'wb') as f:
                    pk.dump({
                        'adj': all_adj,
                        'element': all_element_onehot,
                        'hcount': all_hcount_onehot,
                        'charge': all_charge_onehot,
                        'aromatic': all_aromatic_onehot,
                        'mask': all_mask,
                        'labels': label
                    },f,pk.HIGHEST_PROTOCOL)
        else:
            if explicit_h:
                return all_adj, all_element_onehot, all_charge_onehot, all_aromatic_onehot, label
            else:
                return all_adj, all_element_onehot, all_hcount_onehot , all_charge_onehot, all_aromatic_onehot, label
    else:
        if write == True:
            if explicit_h:
                with open(f'{mode}_data_explicit_h.pkl' if explicit_h else f'{mode}_data.pkl', 'wb') as f:
                    pk.dump({
                        'adj': all_adj,
                        'element': all_element_onehot,
                        'charge': all_charge_onehot,
                        'aromatic': all_aromatic_onehot,
                        'mask': all_mask
                    },f,pk.HIGHEST_PROTOCOL)
            else:
                with open(f'{mode}_data.pkl' if explicit_h else f'{mode}_data.pkl', 'wb') as f:
                    pk.dump({
                        'adj': all_adj,
                        'element': all_element_onehot,
                        'hcount': all_hcount_onehot,
                        'charge': all_charge_onehot,
                        'aromatic': all_aromatic_onehot,
                        'mask': all_mask
                    },f,pk.HIGHEST_PROTOCOL)
        else:
            if explicit_h:
                return all_adj, all_element_onehot, all_charge_onehot, all_aromatic_onehot
            else:
                return all_adj, all_element_onehot, all_hcount_onehot, all_charge_onehot, all_aromatic_onehot


generate_data('validation', explicit_h=True)
generate_data('train', explicit_h=True)
generate_data('test', explicit_h=True)