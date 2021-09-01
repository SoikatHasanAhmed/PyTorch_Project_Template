import os
import json
import argparse

import torch

from trainer.trainer import Trainer
from utils.logger import Logger
from utils.util import get_lr_scheduler
from data_loader import data_loader as module_data
from model import loss as module_loss
from model import metric as module_metric
from model import model as module_arch


def main(config, resume):
    train_logger = Logger()

    # setup data_loader instances
    data_loader_class = getattr(module_data, config['data_loader']['type'])
    data_loader = data_loader_class(**config['data_loader']['args'])
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    model_class = getattr(module_arch, config['model']['type'])
    model = model_class(**config['model']['args'])


    print(model)

    # get function handles of loss and metrics
    loss = {k: getattr(module_loss, v) for k, v in config['loss'].items()}
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer for model and discriminator
    model_trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_class = getattr(torch.optim, config['optimizer']['type'])
    optimizer = dict()
    optimizer['model'] = optimizer_class(model_trainable_params, **config['optimizer']['args'])

    # build learning rate scheduler for model and discriminator
    lr_scheduler = dict()
    lr_scheduler['model'] = get_lr_scheduler(config['lr_scheduler'], optimizer['model'])
    lr_scheduler['discriminator'] = get_lr_scheduler(config['lr_scheduler'],
                                                     optimizer['discriminator'])

    # start to train the network
    trainer = Trainer(config,
                      model,
                      loss,
                      metrics,
                      optimizer,
                      lr_scheduler,
                      resume,
                      data_loader,
                      valid_data_loader,
                      train_logger)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeblurGAN')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        with open(args.config) as handle:
            config = json.load(handle)
        # setting path to save trained models and log files
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with changed configurations.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.yaml', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
