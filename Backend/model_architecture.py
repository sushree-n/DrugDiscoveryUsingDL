import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool as gap

class modelGcn(nn.Module):
    def __init__(self, num_features=5, output_dim=1, dropout=0.2, global_dim=10):
        super(modelGcn, self).__init__()

        # MLPs for GIN layers
        self.mlp1 = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(num_features, num_features * 2),
            nn.ReLU(),
            nn.Linear(num_features * 2, num_features * 2)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(num_features * 2, num_features * 2),
            nn.ReLU(),
            nn.Linear(num_features * 2, num_features * 2)
        )

        # GINConv layers
        self.conv1 = GINConv(self.mlp1)
        self.conv2 = GINConv(self.mlp2)
        self.conv3 = GINConv(self.mlp3)

        # BatchNorm layers
        self.bn1 = nn.BatchNorm1d(num_features)
        self.bn2 = nn.BatchNorm1d(num_features * 2)
        self.bn3 = nn.BatchNorm1d(num_features * 2)

        self.dropout = nn.Dropout(dropout)

        # Deep fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(num_features * 2 + global_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, edge_index, batch, global_x):
        x = self.bn1(F.relu(self.conv1(x, edge_index)))
        x = self.bn2(F.relu(self.conv2(x, edge_index)))
        x = self.bn3(F.relu(self.conv3(x, edge_index)))
        x = gap(x, batch)
        x = self.dropout(x)

        if global_x.dim() == 1 or global_x.shape[0] != x.shape[0]:
            global_x = global_x.view(x.size(0), -1)

        x = torch.cat([x, global_x], dim=1)
        return self.fusion_mlp(x)