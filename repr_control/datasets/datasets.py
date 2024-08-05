from torch.utils.data import Dataset

class SupervisedParkingDataset(Dataset):

    def __init__(self, states, controls):
        self.states = states
        self.controls = controls

    def __len__(self):
        assert len(self.states) == len(self.controls)
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.controls[idx]


class InitialStateFeasibilityDataset(Dataset):

    def __init__(self, states, feasible):
        self.states = states
        self.feasible = feasible

    def __len__(self):
        assert len(self.states) == len(self.feasible)
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.feasible[idx]