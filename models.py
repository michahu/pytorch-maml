import torch
import torch.nn as nn 
import torch.nn.functional as F

class RegressionBase(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)

class Linear(RegressionBase):
    def __init__(self, input_dim, embedding_dim, output_dim):
        super().__init__(input_dim, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_dim)  

    def forward(self, x):
        x.transpose_(1, 0)
        # print(x.shape)
        x = self.embedding(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x

class MLP1(RegressionBase):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__(input_dim, embedding_dim)
        self.l1 = nn.Linear(embedding_dim, hidden_dim)  
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x.transpose_(1, 0)
        x = self.embedding(x)
        x = torch.flatten(x)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x

class MLP2(RegressionBase):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__(input_dim, embedding_dim)
        self.l1 = nn.Linear(embedding_dim, hidden_dim)  
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x.transpose_(1, 0)
        x = self.embedding(x)
        x = torch.flatten(x)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        return x