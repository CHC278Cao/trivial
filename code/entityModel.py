
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

class EmbedModel(nn.Module):
    def __init__(self, emb_size_dict, output_size, emb_dropout, dropout, device):
        super(EmbedModel, self).__init__()

        self.device = device
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_size_dict])
        num_of_embs = sum([y for _, y in emb_size_dict])

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.linear1 = nn.Linear(num_of_embs, 200)
        self.batchNorm1 = nn.BatchNorm1d(200)

        self.linear2 = nn.Linear(200, 200)
        self.batchNorm2 = nn.BatchNorm1d(200)

        self.linear3 = nn.Linear(200, output_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        emb_input = []

        for idx, embedding in enumerate(self.emb_layers):
            x = embedding(inputs[:, idx])
            x = self.emb_dropout(x)
            emb_input.append(x)
        pdb.set_trace()
        emb_input = torch.cat(emb_input, dim=1)
        out = self.batchNorm1(self.linear1(emb_input))
        out = self.dropout(self.activation(out))
        out = self.batchNorm2(self.linear2(out))
        out = self.dropout(self.activation(out))
        out = self.linear3(out)
        out = torch.softmax(out, dim=1)

        return out

    # def _init_weight(self):
    #     for m in self.modules():
    #         for param in m.parameters():
    #             if len(param) >= 2:
    #                 torch.nn.init.kaiming_normal_(param.data)
    #             else:
    #                 torch.nn.init.normal_(param.data)


class entityDataset(Dataset):
    def __init__(self, data, targets = None):
        super(entityDataset, self).__init__()
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data.iloc[idx, :]
        # pdb.set_trace()
        if self.targets is not None:
            target = self.targets.iloc[idx]
            return {
                'data': torch.tensor(data, dtype=torch.long),
                'targets': torch.tensor(target, dtype=torch.long)
            }

        else:
            return {
                'data': torch.tensor(data, dtype=torch.long)
            }









