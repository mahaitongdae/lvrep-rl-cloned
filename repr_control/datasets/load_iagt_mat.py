from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import numpy as np
from datetime import datetime
import torch
import os
keys = ['Xf', 'path', 'actions', 'obs']

class IAGTDataset(object):
    def __init__(self, path):
        pass

def load_iagt_mat(path):
    mat = loadmat(path)
    data = mat['data']
    num_structs = data.shape[1]
    obs_list = []
    actions_list = []
    for i in range(num_structs):
        # x0 = data[0,i]['X0']
        xf = data[0,i]['Xf']
        path = data[0,i]['path']
        actions = data[0,i]['actions']
        flatten_obstacles = data[0,i]['obs'].reshape(1, -1)
        task_info = np.concatenate([xf, flatten_obstacles], axis=1)
        obs = np.concatenate([path[:-1],
                                    np.kron(task_info, np.ones([actions.shape[0], 1]))], axis=1)
        obs_list.append(obs)
        actions_list.append(actions)
    return np.vstack(obs_list), np.vstack(actions_list)


class SupervisedParkingDatasetV2(Dataset):

    def __init__(self, path):
        obs, actions = load_iagt_mat(path)
        self.obs = obs
        self.actions = actions

    def __len__(self):
        assert len(self.obs) == len(self.actions)
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


def test_load_iagt_mat():
    dataset = SupervisedParkingDatasetV2('test.mat')
    print(dataset)

if __name__ == '__main__':
    fname = 'test.mat'
    dataset = SupervisedParkingDatasetV2(fname)
    now = datetime.now()

    # Format date and time
    formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs('./{}'.format(formatted_now), exist_ok=True)
    save_name = fname.split('.')[0]
    torch.save(dataset, './{}'.format(formatted_now) + f'/{save_name}.pt')