import argparse
import os
import cv2
import mmcv
import numpy as np
from mmcv.transforms import Compose
from mmdet.apis import init_detector
from mmengine import Config
from mmengine.registry import TRANSFORMS
from mmengine.structures import InstanceData
from mmrotate.structures import RotatedBoxes
from mmrotate.utils import register_all_modules
from mmrotate.visualization import RotLocalVisualizer
from projects.sam2.build_sam import build_sam2
from projects.sam2.sam2_image_predictor import SAM2ImagePredictor


def parse_args():
    parser = argparse.ArgumentParser(
        'Demo for Zero-shot Oriented Detector with SAM2-AI4RS'
        'Prompted by Predicted HBox of Horizontal Detector',
        add_help=True)
    parser.add_argument('image', type=str, help='path to image file')
    parser.add_argument('det_config', type=str, help='path to det config file')
    parser.add_argument('det_weight', type=str, help='path to det weight file')
    parser.add_argument(
        '--model-cfg',
        type=str,
        default='configs/sam2.1/sam2.1_hiera_l.yaml',
        choices=[
            'configs/sam2.1/sam2.1_hiera_t.yaml',
            'configs/sam2.1/sam2.1_hiera_s.yaml',
            'configs/sam2.1/sam2.1_hiera_b+.yaml',
            'configs/sam2.1/sam2.1_hiera_l.yaml'],
        help='model cfg')
    parser.add_argument(
        '--sam-weight',
        type=str,
        default='./work_dirs/sam2.1_ai4rs/sam2.1_hiera_large.pt',
        help='path to checkpoint file')
    parser.add_argument(
        '--out-path', '-o', type=str, default='output.png', help='output path')
    parser.add_argument(
        '--box-thr', '-b', type=float, default=0.3, help='box threshold')
    parser.add_argument(
        '--max-batch-num-pred',
        type=int,
        default=100,
        help='max prediction number of mask generation (avoid OOM)')
    parser.add_argument('--set-min-box', action='store_true')
    parser.add_argument('--result-with-mask', action='store_true')
    parser.add_argument(
        '--device', '-s', default='cuda:0', help='Device used for inference')
    parser.add_argument('--save-masks', type=bool, default=False, help='save output masks of SAM')
    parser.add_argument('--masks-path', type=str, default='./mmrotate_sam_output_masks/',
                        help='the save path of the mask output by sam')
    return parser.parse_args()


def mask2rbox(mask):
    y, x = np.nonzero(mask)
    points = np.stack([x, y], axis=-1)
    (cx, cy), (w, h), a = cv2.minAreaRect(points)
    r_bbox = np.array([cx, cy, w, h, a / 180 * np.pi])
    return r_bbox


def get_instancedata_resultlist(r_bboxes,
                                labels,
                                masks,
                                scores,
                                result_with_mask=False):
    results = InstanceData()
    results.bboxes = RotatedBoxes(r_bboxes)
    results.scores = scores
    results.labels = labels
    if result_with_mask:
        results.masks = masks.cpu().numpy()
    results_list = [results]
    return results_list


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)

    # prepare the detector model
    detector_cfg = Config.fromfile(args.det_config)
    if 'init_cfg' in detector_cfg.model.backbone:
        detector_cfg.model.backbone.init_cfg = None
    if args.set_min_box:
        detector_cfg.model.test_cfg['min_bbox_size'] = 10
    det_model = init_detector(
        detector_cfg, args.det_weight, device='cpu', cfg_options={})
    det_model = det_model.to(args.device)

    # prepare the sam model
    sam_model = SAM2ImagePredictor(build_sam2(args.model_cfg, args.sam_weight))
    sam_model.model = sam_model.model.to(args.device)

    # load image
    naive_test_pipeline = [
        dict(type='mmdet.LoadImageFromFile'),
        dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    ]
    tfm = Compose([TRANSFORMS.build(p) for p in naive_test_pipeline])
    data = tfm(dict(img_path=args.image))
    for k, v in data.items():
        data[k] = [v.to(args.device)]

    # predict with detector
    pred_results = det_model.test_step(data)
    pred_bboxes = pred_results[0].pred_instances.bboxes
    # If horizontal detector is used, directly use the predicted HBB as
    # prompts. If oriented detector is used, the OBB is converted to HBB.
    if pred_bboxes.size(-1) == 5:
        pred_r_bboxes = RotatedBoxes(pred_bboxes)
        h_bboxes = pred_r_bboxes.convert_to('hbox').tensor
    elif pred_bboxes.size(-1) == 4:
        h_bboxes = pred_bboxes
    else:
        raise ValueError(f'The dimension of box is {pred_bboxes.size(-1)}.')
    labels = pred_results[0].pred_instances.labels
    scores = pred_results[0].pred_instances.scores
    keep = scores > args.box_thr
    h_bboxes = h_bboxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    # prompt sam with predicted boxes
    img = mmcv.imread(args.image, channel_order='RGB')
    sam_model.set_image(img)
    # Too many predictions may result in OOM, hence,
    # we process the predictions in multiple batches.
    masks = []
    N = args.max_batch_num_pred
    num_pred = len(h_bboxes)
    num_batches = int(np.ceil(num_pred / N))
    for i in range(num_batches):
        left_index = i * N
        right_index = (i + 1) * N
        if i == num_batches - 1:
            batch_boxes = h_bboxes[left_index:]
        else:
            batch_boxes = h_bboxes[left_index:right_index]

        batch_masks, _, _ = sam_model.predict(
            point_coords=None,
            point_labels=None,
            box=batch_boxes,
            multimask_output=False,
        )
        batch_masks = batch_masks.squeeze(1)
        masks.extend([m for m in batch_masks])

    masks = np.array(masks)
    r_bboxes = np.array([mask2rbox(mask) for mask in masks], dtype=np.float32)
    results_list = get_instancedata_resultlist(r_bboxes, labels, masks, scores,
                                               args.result_with_mask)

    # initialize visualizer
    visualizer = RotLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')], name='SAM2 AI4RS')
    out_img = visualizer._draw_instances(img, results_list[0],
                                         det_model.dataset_meta['classes'],
                                         det_model.dataset_meta['palette'])
    mmcv.imwrite(out_img, args.out_path)

    # save masks
    if args.save_masks:
        base_name = os.path.basename(args.image)
        file_name, ext = os.path.splitext(base_name)
        os.makedirs(args.masks_path, exist_ok=True)

        for i in range(masks.shape[0]):
            mask = (masks[i] > 0).astype(np.uint8) * 255

            # save black and white masks
            out_path = os.path.join(args.masks_path, f"{file_name}_mask_{i}{ext}")
            cv2.imwrite(out_path, mask)

            # save color masks
            color = (0, 0, 255)
            alpha = 0.3
            colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            colored_mask[mask > 0] = color
            overlay = img.copy()
            overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
            overlay_save_path = os.path.join(args.masks_path, f"{file_name}_overlay_{i}.png")
            mmcv.imwrite(overlay, overlay_save_path)

        print(f"Saved {masks.shape[0]} masks to {args.masks_path}")


if __name__ == '__main__':
    main()