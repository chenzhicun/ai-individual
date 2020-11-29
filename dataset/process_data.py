import os
import pandas as pd
import numpy as np
import pickle as pk
import torch
os.chdir('../data/validation')


def load_data(filefolder):
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()
    data = data['onehots']
    label = pd.read_csv(os.path.abspath(filefolder + '/names_labels.txt'), sep=',')
    label = label['Label'].values
    data=torch.from_numpy(data)
    print(data.shape)
    label=torch.from_numpy(label)
    return data, label


data, label = load_data('./')
with open('validation_data.pkl', 'wb') as f:
    pk.dump({
        'data':data,
        'label':label
    },f,pk.HIGHEST_PROTOCOL)