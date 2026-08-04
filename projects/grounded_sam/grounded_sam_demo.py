import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import argparse
import numpy as np
import nltk
from mmengine import Config
from mmengine.structures import InstanceData
import mmcv
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes
from mmdet.visualization import DetLocalVisualizer
from projects.segment_anything import sam_model_registry, SamPredictor


def parse_args():
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument('image', type=str, help='path to image file')
    parser.add_argument('det_config', type=str, help='path to det config file')
    parser.add_argument('det_weight', type=str, help='path to det weight file')
    # Once you input a format similar to $: xxx, it indicates that
    # the prompt is based on the dataset class name.
    # support $: coco, $: voc, $: cityscapes, $: lvis, $: imagenet_det.
    # detail to `mmdet/evaluation/functional/class_names.py`
    parser.add_argument(
        '--texts', type=str, default='$: coco',
        help='text prompt, such as "bench . car .", "$: coco"')
    parser.add_argument(
        '--sam-type',
        type=str,
        default='vit_h',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help='sam type')
    parser.add_argument(
        '--sam-weight',
        type=str,
        default='../models/sam_vit_h_4b8939.pth',
        help='path to checkpoint file')
    parser.add_argument('--nltk_root', type=str, default='./work_dirs/nltk_data')
    parser.add_argument(
        '--out-path', '-o', type=str, default='output.png', help='output path')
    parser.add_argument(
        '--box-thr', '-b', type=float, default=0.1, help='box threshold')
    parser.add_argument(
        '--max-batch-num-pred',
        type=int,
        default=100,
        help='max prediction number of mask generation (avoid OOM)')
    parser.add_argument('--result-with-mask', type=bool, default=True)
    parser.add_argument(
        '--device', '-s', default='cuda:0', help='Device used for inference')
    return parser.parse_args()


def download_nltk_data(nltk_root):

    tasks = [
        ('tokenizers', 'punkt.zip',
         'https://gh.chjina.com/https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip'),
        ('tokenizers', 'punkt_tab.zip',
         'https://gh.chjina.com/https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip'),
        ('taggers', 'averaged_perceptron_tagger.zip',
         'https://gh.chjina.com/https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip'),
        ('taggers', 'averaged_perceptron_tagger_eng.zip',
         'https://gh.chjina.com/https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip')
    ]

    for sub_dir, zip_name, url in tasks:
        target_dir = os.path.join(nltk_root, sub_dir)
        folder_name = zip_name.replace('.zip', '')
        final_path = os.path.join(target_dir, folder_name)

        if not os.path.exists(final_path):
            print(f"[*] Downloading and extracting {zip_name}...")
            os.makedirs(target_dir, exist_ok=True)
            zip_path = os.path.join(target_dir, zip_name)
            os.system(f"wget -O {zip_path} {url}")
            os.system(f"unzip -o {zip_path} -d {target_dir}")
            os.system(f"rm {zip_path}")
        else:
            print(f"[+] {folder_name} already exists. Skipping.")

    if nltk_root not in nltk.data.path:
        nltk.data.path.insert(0, nltk_root)

    def mock_download(info_or_id=None, download_dir=None, *args, **kwargs):
        print(f"[*] NLTK network access has been disabled. Using local data {info_or_id}.")
        return True
    nltk.download = mock_download

def get_instancedata_resultlist(bboxes,
                                labels,
                                masks,
                                scores,
                                result_with_mask=False):
    results = InstanceData()
    results.bboxes = bboxes.cpu().numpy()
    results.scores = scores.cpu().numpy()
    results.labels = labels.cpu().numpy()
    if result_with_mask:
        results.masks = masks.cpu().numpy()
    results_list = [results]
    return results_list
    
def main():
    args = parse_args()

    download_nltk_data(args.nltk_root)

    det_config = Config.fromstring(f'_base_=[\'{args.det_config}\']', file_format='.py')
    det_model = DetInferencer(
        det_config, args.det_weight, device=args.device)

    # predict with detector
    texts = args.texts
    if texts.startswith('$:'):
        dataset_name = texts[3:].strip()
        class_names = get_classes(dataset_name)
        texts = [tuple(class_names)]
    pred_results = det_model(args.image, texts=texts)
    bboxes = torch.tensor(pred_results['predictions'][0]['bboxes'], device=args.device)
    labels = torch.tensor(pred_results['predictions'][0]['labels'], device=args.device)
    scores = torch.tensor(pred_results['predictions'][0]['scores'], device=args.device)
    keep = scores > args.box_thr
    bboxes = bboxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    # prepare the sam model
    build_sam = sam_model_registry[args.sam_type]
    sam_model = SamPredictor(build_sam(checkpoint=args.sam_weight))
    sam_model.model = sam_model.model.to(args.device)

    # prompt sam with predicted boxes
    img = mmcv.imread(args.image, channel_order='RGB')
    sam_model.set_image(img, image_format='RGB')
    # Too many predictions may result in OOM, hence,
    # we process the predictions in multiple batches.
    masks = []
    N = args.max_batch_num_pred
    num_pred = len(bboxes)
    num_batches = int(np.ceil(num_pred / N))
    for i in range(num_batches):
        left_index = i * N
        right_index = (i + 1) * N
        if i == num_batches - 1:
            batch_boxes = bboxes[left_index:]
        else:
            batch_boxes = bboxes[left_index:right_index]

        transformed_boxes = sam_model.transform.apply_boxes_torch(
            batch_boxes, img.shape[:2])
        batch_masks = sam_model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False)[0]
        batch_masks = batch_masks.squeeze(1).cpu()
        masks.extend([*batch_masks])
    masks = torch.stack(masks, dim=0)

    results_list = get_instancedata_resultlist(bboxes, labels, masks, scores,
                                               args.result_with_mask)

    # initialize visualizer
    visualizer = DetLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')], name='Grounded-SAM')
    out_img = visualizer._draw_instances(img, results_list[0],
                                         classes=None, palette=None)
    mmcv.imwrite(out_img, args.out_path)

if __name__ == '__main__':
    main()