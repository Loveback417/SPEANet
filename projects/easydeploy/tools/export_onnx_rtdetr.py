import argparse
import os
from io import BytesIO
import onnx
import torch
from mmengine.logging import print_log
from mmengine.utils.path import mkdir_or_exist
from mmdet.apis import init_detector

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument(
        '--work-dir', default='./work_dir', help='Path to save export model')
    parser.add_argument(
        '--img-size',
        nargs=2,
        type=int,
        default=[640, 640],
        help='Image size of height and width')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify onnx model by onnx-sim')
    parser.add_argument(
        '--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument(
        '--dynamic-batch',
        action='store_true',
        help='Export the model to support dynamic batch size.'
    )
    parser.add_argument(
        '--dynamic-hw',
        action='store_true',
        help='Export the model to support dynamic height and width (requires dynamic batch).'
    )
    parser.add_argument(
        '--verbose', type=bool, default=False, help='Enable verbose logging')
    args = parser.parse_args()
    return args


def build_model_from_cfg(config_path, checkpoint_path, device):
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model


def main(args):
    target_device = args.device
    if 'cuda' in target_device and not torch.cuda.is_available():
        raise TypeError(f'CUDA device {target_device} is not available.')

    mkdir_or_exist(args.work_dir)

    print_log(f'Building model from config: {args.config} and checkpoint: {args.checkpoint}', logger='current')
    baseModel = build_model_from_cfg(args.config, args.checkpoint, args.device)
    deploy_model = baseModel
    deploy_model.eval()

    H, W = args.img_size[0], args.img_size[1]
    fake_input = torch.randn(args.batch_size, 3, H, W).to(target_device)

    input_names = ['images']
    output_names = ['scores', 'boxes']

    dynamic_axes = None
    if args.dynamic_batch or args.dynamic_hw:
        dynamic_axes = {
            'images': {0: 'batch_size'},
            'scores': {0: 'batch_size'},
            'boxes': {0: 'batch_size'}
        }
        if args.dynamic_hw:
            # 2: Height (H), 3: Width (W)
            dynamic_axes['images'].update({2: 'height', 3: 'width'})
            print_log("Enabling dynamic H/W. The model must support variable input size.", logger='current')
        print_log(f"Dynamic axes configured: {dynamic_axes}", logger='current')

    with torch.no_grad():
        deploy_model(fake_input)

    save_onnx_path = os.path.join(
        args.work_dir,
        os.path.basename(args.checkpoint).replace('pth', 'onnx'))
    print_log(f'Starting ONNX export to temporary buffer (Opset: {args.opset})...', logger='current')

    # export onnx
    with BytesIO() as f:
        torch.onnx.export(
            deploy_model,
            fake_input,
            f,
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            dynamic_axes=dynamic_axes,
            verbose=args.verbose)
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
        print_log('ONNX model check successful. Model structure is valid.', logger='current')

    if args.simplify:
        try:
            import onnxsim
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print_log(f'Simplify failure: {e}')
    onnx.save(onnx_model, save_onnx_path)
    print_log(f'ONNX export success, save into {save_onnx_path}')


if __name__ == '__main__':
    args = parse_args()
    args.config = '../../rtdetr/configs/rtdetr_r50vd_8xb2_72e_coco.py'
    args.checkpoint = '../../../rtdetr_r50vd_8xb2-72e_coco_ad2bdcfe.pth'
    args.work_dir = '../../../work_dirs/easydeploy/rtdetr'
    args.img_size = [640, 640]
    args.batch_size = 1
    args.simplify = True
    main(args)