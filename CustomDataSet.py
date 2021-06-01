import torch


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, domain):
        self.domain = domain

    def __len__(self):
        return len(self.domain)

    def __getitem__(self, index):
        return self.domain[index]
