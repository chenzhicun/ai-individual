from os import name
import torch
import numpy as np
import argparse
import os
from collections import OrderedDict
import torch.utils.data as Data
from tqdm import tqdm


def generate_dataloader():
    data = np.load('./data/test/names_onehots.npy', allow_pickle=True).item()
    name = data['names']
    data = data['onehots']
    data = torch.from_numpy(data)
    data = data.unsqueeze(1)
    # data = data.float().to(torch.device("cuda"))
    print(data.shape)
    # pred_dataset = Data.TensorDataset(data)
    # pred_sampler = Data.SequentialSampler(pred_dataset)
    # pred_datalorder = Data.DataLoader(pred_dataset, sampler=pred_sampler, batch_size=64)    
    
    return data, name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Drug Molecular Toxicity Predicting')
    parser.add_argument('--model', default=None, help="the model name")
    parser.add_argument('--exp_path', default=None, help="the path to model's saved state dict")
    opt = parser.parse_args()
    exp_id = opt.exp_path.split('/')[-1]
    if opt.model == "resnet10":
        import model.Resnet.resnet_model as rm
        m = rm.resnet10()
        m.to(torch.device("cuda"))
        m = torch.nn.DataParallel(m)
    if opt.model == 'resnet18':
        import model.Resnet.resnet_model as rm
        m = rm.resnet18()
        m.to(torch.device("cuda"))
        m = torch.nn.DataParallel(m)
    if opt.model == 'resnet34':
        import model.Resnet.resnet_model as rm
        m = rm.resnet34()
        m.to(torch.device("cuda"))
        m = torch.nn.DataParallel(m)
    elif opt.model == None:
        print('please give model name!')
        quit()
    if opt.exp_path == None:
        print('please give the experiment path!')
        quit()
    else:
        saved_state_path = f'{opt.exp_path}/{exp_id}_best.pth'
        state_dict = torch.load(saved_state_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = 'module.' + k
            new_state_dict[k] = v
        m.load_state_dict(new_state_dict)
        # m.load_state_dict(state_dict)

    m.eval()
    all_outputs = []
    data, name = generate_dataloader()
    with torch.no_grad():
        data = data.float().to(torch.device('cuda'))
        outputs = m(data)
        outputs = outputs[:, 1].detach().cpu().numpy()
        for i in range(len(outputs)):
            all_outputs.append(outputs[i])
    
    with open(f'{opt.exp_path}/output_518030910173.txt', 'w') as f:
        f.write('Chemical,Label\n')
        for i in range(len(all_outputs)):
            line = f'{name[i]},{all_outputs[i]}\n'
            f.write(line)

    with open('output_518030910173.txt', 'w') as f:
        f.write('Chemical,Label\n')
        for i in range(len(all_outputs)):
            line = f'{name[i]},{all_outputs[i]}\n'
            f.write(line)
