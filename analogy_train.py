import torch
import math
import os
import time
import json
import logging
import wandb
import random 
import numpy as np

import torch.nn as nn

from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.modules import MetaLinear
from torchmeta.transforms import ClassSplitter
from meta_dataset import Analogy
# from models import Linear, MLP1, MLP2

from maml.model import MetaMLPModel
from maml.metalearners import ModelAgnosticMetaLearning

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    wandb.init(project='meta-analogy')
    # print(args.use_cuda)
    # print(f'CUDA IS AVAILABLE: {torch.cuda.is_available()}')
    # assert args.use_cuda
    # assert torch.cuda.is_available()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    if (args.output_folder is not None):
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
            logging.debug('Creating folder `{0}`'.format(args.output_folder))

        folder = os.path.join(args.output_folder,
                              time.strftime('%Y-%m-%d_%H%M%S'))
        os.makedirs(folder)
        logging.debug('Creating folder `{0}`'.format(folder))

        # args.folder = os.path.abspath(args.folder)
        args.model_path = os.path.abspath(os.path.join(folder, 'model.th'))
        # Save the configuration in a config.json file
        with open(os.path.join(folder, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        logging.info('Saving configuration file in `{0}`'.format(
                     os.path.abspath(os.path.join(folder, 'config.json'))))

    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_shots,
                                      num_test_per_class=args.num_shots_test)

    # meta_train_dataset = Analogy(num_samples_per_task=args.batch_size,
    #                         dataset_transform=dataset_transform)
    # meta_val_dataset = Analogy(num_samples_per_task=args.batch_size,
    #                         dataset_transform=dataset_transform)
    meta_train_dataset = Analogy(dataset_transform=dataset_transform)
    meta_val_dataset = Analogy(dataset_transform=dataset_transform)

    meta_train_dataloader = BatchMetaDataLoader(meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
    meta_val_dataloader = BatchMetaDataLoader(meta_val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    if args.model_type == 'linear':
        model = MetaLinear(in_features=300, out_features=300)
    elif args.model_type == 'mlp1':
        model = MetaMLPModel(in_features=300, out_features=300, hidden_sizes=[500])
    elif args.model_type == 'mlp2':
        model = MetaMLPModel(in_features=300, out_features=300, hidden_sizes=[500, 500])
    else:
        raise ValueError('unrecognized model type')

    loss_function = nn.MSELoss()
    wandb.watch(model)

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    metalearner = ModelAgnosticMetaLearning(model,
                                            meta_optimizer,
                                            first_order=args.first_order,
                                            num_adaptation_steps=args.num_steps,
                                            step_size=args.step_size,
                                            loss_function=loss_function,
                                            device=device)

    best_value = None

    # Training loop
    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
    for epoch in range(args.num_epochs):
        metalearner.train(meta_train_dataloader,
                          max_batches=args.num_batches,
                          verbose=args.verbose,
                          desc='Training',
                          leave=False)
        results = metalearner.evaluate(meta_val_dataloader,
                                       max_batches=args.num_batches,
                                       verbose=args.verbose,
                                       desc=epoch_desc.format(epoch + 1))
        wandb.log({'results' : results})

        # Save best model
        if 'accuracies_after' in results:
            if (best_value is None) or (best_value < results['accuracies_after']):
                best_value = results['accuracies_after']
                save_model = True
        elif (best_value is None) or (best_value > results['mean_outer_loss']):
            best_value = results['mean_outer_loss']
            save_model = True
        else:
            save_model = False

        if save_model and (args.output_folder is not None):
            with open(args.model_path, 'wb') as f:
                torch.save(model.state_dict(), f)

    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

    if hasattr(meta_train_dataset, 'close'):
        meta_train_dataset.close()
        meta_val_dataset.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')

    # General
    # parser.add_argument('folder', type=str,
    #     help='Path to the folder the data is downloaded to.')
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed for reproducibility')
    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder to save the model.')
    parser.add_argument('--num-shots', type=int, default=8,
        help='Number of training example per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-shots-test', type=int, default=16,
        help='Number of test example per class. If negative, same as the number '
        'of training examples `--num-shots` (default: 15).')

    # Model
    parser.add_argument('--model-type', type=str, default='linear',
        help='choose amongst linear, mlp1, and mlp2')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels in each convolution layer of the VGG network '
        '(default: 64).')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=64,
        help='Number of tasks in a batch of tasks (default: 25).')
    parser.add_argument('--num-steps', type=int, default=1,
        help='Number of fast adaptation steps, ie. gradient descent '
        'updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=50,
        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batch of tasks per epoch (default: 100).')
    parser.add_argument('--step-size', type=float, default=0.1,
        help='Size of the fast adaptation step, ie. learning rate in the '
        'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first order approximation, do not use higher-order '
        'derivatives during meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
        help='Learning rate for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is Adam (default: 1e-3).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')

    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    main(args)
