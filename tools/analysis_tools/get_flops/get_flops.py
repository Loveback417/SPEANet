# Copyright (c) ai4rs. All rights reserved.
import argparse
from pathlib import Path
import torch

from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmrotate.utils import register_all_modules
from mmrotate.registry import MODELS
from flops_counter import calculate_flops


def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[640, 640],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    return parser.parse_args()


def inference(args, logger):
    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    config = args.config
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    init_default_scope(config.get('default_scope', 'ai4rs'))

    # model
    model = MODELS.build(config.model)
    if torch.cuda.is_available():
        model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid input shape')
    
    # input tensor
    # automatically generate a input tensor with the given input_shape.
    data_samples = DetDataSample()
    data_samples.set_metainfo(dict(img_shape=(h, w)))
    data_batch = {'inputs': [torch.rand(3, h, w)], 'data_samples': [data_samples]}
    data = model.data_preprocessor(data_batch)
    result = {'ori_shape': (h, w), 'pad_shape': data['inputs'].shape[-2:]}
    flops, macs, params = calculate_flops(model=model,
                                          kwargs=dict(
                                              inputs=data['inputs'],
                                              data_samples=data['data_samples']),
                                          output_as_string=True,
                                          output_precision=4,
                                          print_detailed=False)
    result['flops'] = flops
    result['params'] = params
    result['macs'] = macs

    return result


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules_mmdet()
    register_all_modules()

    logger = MMLogger.get_instance(name='MMLogger')
    result = inference(args, logger)
    split_line = '=' * 30
    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    flops = result['flops']
    params = result['params']
    macs = result['macs']

    if pad_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {pad_shape}')
    print(f'{split_line}\n'
          # f'Compute type: {compute_type}\n'
          f'Input shape: {pad_shape}\n'
          f'Flops: {flops}\n'
          f'Params: {params}\n'
          f'MACs: {macs}\n'
          f'{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify '
          'that the flops computation is correct.')


if __name__ == '__main__':
    main()