import torch
import torch.nn as nn 
import torch.nn.functional as F

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)

class RegressionBase(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embeddings,
            freeze=True
            )

class Linear(RegressionBase):
    def __init__(self, embeddings, input_dim, output_dim):
        super().__init__(embeddings)
        self.linear = nn.Linear(embedding_dim, output_dim)  
        #TODO: finish fixing this.

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

class MetaMLPModel(MetaModule):
    """Multi-layer Perceptron architecture from [1].

    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of classes (output of the model).

    hidden_sizes : list of int
        Size of the intermediate representations. The length of this list
        corresponds to the number of hidden layers.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_features, out_features, hidden_sizes):
        super(MetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        layer_sizes = [in_features] + hidden_sizes
        self.features = MetaSequential(OrderedDict([('layer{0}'.format(i + 1),
            MetaSequential(OrderedDict([
                ('linear', MetaLinear(hidden_size, layer_sizes[i + 1], bias=True)),
                ('relu', nn.ReLU())
            ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.classifier = MetaLinear(hidden_sizes[-1], out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits