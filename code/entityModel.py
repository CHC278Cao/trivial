
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

class EmbedModel(nn.Module):
    def __init__(self, emb_size_dict, output_size, emb_dropout, dropout, device):
        super(EmbedModel, self).__init__()

        self.device = device
        self.target_cols, self.emb_size_input = zip(*emb_size_dict.items())
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in self.emb_size_input])
        num_of_embs = sum([y for _, y in self.emb_size_input])

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.linear1 = nn.Linear(num_of_embs, 300)


        self.linear2 = nn.Linear(300, output_size)

        self.batchNorm = nn.BatchNorm1d(300)
        self.droput = nn.Dropout(dropout)

    def forward(self, inputs):
        emb_input = []

        for idx, embedding in enumerate(self.emb_layers):
            x = embedding(inputs[self.target_cols[idx]].to(self.device))
            x = self.emb_dropout(x)
            emb_input.append(x)
        # pdb.set_trace()
        emb_input = torch.cat(emb_input, dim=1)
        out = self.linear1(emb_input)
        out = self.batchNorm(out)
        out = torch.sigmoid(self.linear2(out))

        return out

    def _init_weight(self):
        for m in self.modules():
            for param in m.parameters():
                if len(param) >= 2:
                    torch.nn.init.kaiming_normal_(param.data)
                else:
                    torch.nn.init.normal_(param.data)


class entityDataset(Dataset):
    def __init__(self, df, cat_cols, targets = None):
        super(entityDataset, self).__init__()
        self.df = df
        self.cat_cols = cat_cols

        self.targets = targets
        if targets is not None:
            self.targets_values = self.df[targets].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data_dict = {}
        for c in self.cat_cols:
            feature_data = self.df[c].values
            data = torch.tensor(feature_data[idx], dtype=torch.long)
            data_dict[c] = data
        if self.targets is not None:
            data_dict[self.targets] = torch.tensor(self.targets_values[idx], dtype=torch.long)


        return data_dict







