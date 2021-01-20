from typing import Text
from torchmeta.utils.data import Task, MetaDataset
import torchtext
import torch
import torch.nn as nn
import numpy as np
# import spacy
import os, random

class Analogy(MetaDataset):
    def __init__(self, num_tasks=99, target_transform=None, dataset_transform=None):
        super(Analogy, self).__init__(meta_split='train',
            target_transform=target_transform,
            dataset_transform=dataset_transform)

        # self.num_samples_per_task = num_samples_per_task
        self.num_tasks = num_tasks
        self.inputs, self.targets = self.load_data()

        # print(len(self.inputs), len(self.targets), num_tasks)
        assert len(self.inputs) == len(self.targets) == num_tasks

        self.vocab = torchtext.vocab.GloVe(name='840B', dim=300)
        pretrained_embeddings = self.vocab.vectors

        self.encoder = nn.Embedding.from_pretrained(pretrained_embeddings)        

    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, index):
        task = AnalogyTask(index, self.inputs[index], self.targets[index], self.encoder, self.vocab.stoi)

        if self.dataset_transform is not None:
            task = self.dataset_transform(task)
        
        return task

    def load_data(self):
        all_inputs, all_targets = [], []
        path = str(os.getcwd()) + '/Meta-Training/'
        for filename in os.listdir(path):
            with open(path+filename) as f:
                lines = f.read().splitlines()

                task_inputs, task_targets = [], []
                for line in lines:
                    inp, targ = line.split(',')
                    task_inputs.append(inp)
                    task_targets.append(targ)

                all_inputs.append(task_inputs)
                all_targets.append(task_targets)

        return all_inputs, all_targets


class AnalogyTask(Task):
    def __init__(self, index, inputs, targets, encoder, word_to_idx):
        super(AnalogyTask, self).__init__(index, None) # Regression Task
        assert len(inputs) == len(targets)
        self.num_samples = len(inputs)
        # print(index)
        # print(len(inputs[index]), self.num_samples)

        # indices = np.random.permutation(len(inputs))[:self.num_samples+1]
        # raw_idx = np.take(inputs, indices)
        # raw_targ = np.take(targets, indices)
        tokenized_idx = []
        tokenized_targ = [] 
        #for idx, targ in zip(raw_idx, raw_targ):
        for idx, targ in zip(inputs, targets):
            tokenized_idx.append(word_to_idx[idx])
            tokenized_targ.append(word_to_idx[targ])
        # print(raw_idx[0], raw_targ[0])
        # print(raw_idx, raw_targ)
        # tokenized_idx = torch.Tensor(tokenized_idx, dtype=torch.long)
        # tokenized_targ = torch.Tensor(tokenized_targ, dtype=torch.long)
        tokenized_idx = torch.tensor(tokenized_idx)
        tokenized_targ = torch.tensor(tokenized_targ)
        # print(tokenized_idx, tokenized_targ)
        self._inputs = encoder(tokenized_idx)
        self._targets = encoder(tokenized_targ)
        # print(self._inputs.size())
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        input, target = self._inputs[index], self._targets[index]
        return (input, target)