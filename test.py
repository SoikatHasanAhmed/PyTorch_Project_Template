"""
Test file for the network
"""
import os
import argparse

import torch
from tqdm import tqdm
import numpy as np


def main(resume):
    # load checkpoint
    checkpoint = torch.load(resume)
    config = checkpoint['config']

    # setup data_loader instances
    data_loader_class = getattr(module_data, config['data_loader']['type'])
    data_loader_config_args = {
        "data_dir": config['data_loader']['args']['data_dir'],
        'batch_size': 16,  # use large batch_size
        'shuffle': False,  # do not shuffle
        'validation_split': 0.0,  # do not split, just use the full dataset
        'num_workers': 16  # use large num_workers
    }
    data_loader = data_loader_class(**data_loader_config_args)

    # build model architecture
    model_class = getattr(module_arch, config['model']['type'])
    model = model_class(**config['model']['args'])

    # get function handles of loss and metrics
    loss_fn = {k: getattr(module_loss, v) for k, v in config['loss'].items()}
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # prepare model for testing
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.load_state_dict(checkpoint['model'])

    model.eval()

    total_loss = 0.0
    total_metrics = np.zeros(len(metric_fns))

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(data_loader, ascii=True)):
            blurred = sample['blurred'].to(device)
            sharp = sample['sharp'].to(device)

            deblurred = model(blurred)

            denormalized_deblurred = denormalize(deblurred)
            denormalized_sharp = denormalize(sharp)

            # computing loss, metrics on test set
            content_loss_lambda = config['others']['content_loss_lambda']
            adversarial_loss_fn = loss_fn['adversarial']
            content_loss_fn = loss_fn['content']
            loss = content_loss_fn(deblurred, sharp) * content_loss_lambda

            total_loss += loss.item()
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(denormalized_deblurred, denormalized_sharp)

    n_samples = len(data_loader)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__:
                    total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeblurGAN')

    parser.add_argument('-r', '--resume',
                        required=True,
                        type=str,
                        help='path to latest checkpoint')
    parser.add_argument('--device',
                        default=None,
                        type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    import data_loader.data_loader as module_data
    import model.loss as module_loss
    import model.metric as module_metric
    import model.model as module_arch
    from utils.util import denormalize

    main(args.resume)
