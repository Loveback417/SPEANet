import os
from argparse import ArgumentParser
import mmcv
import torch
from mmengine.config import Config, ConfigDict
from mmengine.utils import ProgressBar, path
from mmcv.transforms import Compose
from mmdet.utils import get_test_pipeline_cfg
from mmdet.utils.misc import get_file_list
from mmdet.structures import DetDataSample
from mmdet.registry import MODELS, VISUALIZERS
from projects.easydeploy.model import ORTWrapper, TRTWrapper

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--img', help='Image path, include image file, dir and URL.')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args

def preprocess(config):
    data_preprocess = config.get('model', {}).get('data_preprocessor', {})
    mean = data_preprocess.get('mean', [0., 0., 0.])
    std = data_preprocess.get('std', [1., 1., 1.])
    mean = torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)

    class PreProcess(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x[None].float()
            x -= mean.to(x.device)
            x /= std.to(x.device)
            return x

    return PreProcess().eval()


def main(args):
    # build the model from a config file and a checkpoint file
    if args.checkpoint.endswith('.onnx'):
        model = ORTWrapper(args.checkpoint, args.device)
    elif args.checkpoint.endswith('.engine') or args.checkpoint.endswith(
            '.plan'):
        model = TRTWrapper(args.checkpoint, args.device)
    else:
        raise NotImplementedError
    model.to(args.device)

    cfg = Config.fromfile(args.config)
    files, source_type = get_file_list(args.img)

    test_pipeline = get_test_pipeline_cfg(cfg)
    test_pipeline[0] = ConfigDict({'type': 'mmdet.LoadImageFromNDArray'})
    test_pipeline = Compose(test_pipeline)
    pre_pipeline = preprocess(cfg)
    bbox_head = MODELS.build(cfg.model.bbox_head)

    # init visualizer
    visualizer = VISUALIZERS.build(cfg.visualizer)

    if not args.show:
        path.mkdir_or_exist(args.out_dir)

    # start detector inference
    progress_bar = ProgressBar(len(files))
    for i, file in enumerate(files):
        bgr = mmcv.imread(file)
        rgb = mmcv.imconvert(bgr, 'bgr', 'rgb')
        data, samples = test_pipeline(dict(img=rgb, img_id=i)).values()
        data = pre_pipeline(data).to(args.device)
        outputs = model(data)
        cls_score, bbox_pred = outputs
        results = bbox_head._predict_by_feat_single(cls_score[0], bbox_pred[0], samples.metainfo)
        det_data_sample = DetDataSample()
        det_data_sample.pred_instances = results

        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        visualizer.add_datasample(
            'result',
            rgb,
            data_sample=det_data_sample,
            draw_gt=False,
            show=out_file is None,
            wait_time=0,
            out_file=out_file,
            pred_score_thr=args.score_thr)
        progress_bar.update()


if __name__ == '__main__':
    args = parse_args()
    args.img = '../../../demo/det.jpg'
    args.config = '../../rtdetr/configs/rtdetr_r50vd_8xb2_72e_coco.py'
    # args.checkpoint = '../../../work_dirs/easydeploy/rtdetr/rtdetr_r50vd_8xb2-72e_coco_ad2bdcfe.onnx'
    args.checkpoint = '../../../work_dirs/easydeploy/rtdetr/rtdetr_r50vd_8xb2-72e_coco_ad2bdcfe.engine'
    args.out_dir = '../../../work_dirs/easydeploy/rtdetr'
    # args.device = 'cpu'
    args.device = 'cuda:0'
    main(args)
