import argparse
import json
import os
import cv2
import numpy as np
import concurrent.futures
from natsort import natsorted
from tqdm import tqdm
import instances2dict_with_polygons as cs


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)

    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[0::2]) for p in poly)
        x1 = max(max(p[0::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]

    return boxes_from_polys


def process_single_image(inst_file, ann_dir, cat_name_to_id):
    base_name = inst_file.replace('_instance_id_RGB.png', '')
    img_filename = base_name + '.png'
    seg_file = base_name + '_instance_color_RGB.png'

    inst_fullname = os.path.join(ann_dir, inst_file)
    seg_fullname = os.path.join(ann_dir, seg_file)

    if not os.path.exists(seg_fullname):
        return None

    objects = cs.instances2dict_with_polygons([seg_fullname], [inst_fullname], verbose=False)

    temp_annotations = []

    tmp_img = cv2.imread(inst_fullname)
    if tmp_img is None: return None
    h, w = tmp_img.shape[:2]

    for _, instance_list in objects.items():
        for cls_name, objs in instance_list.items():
            if cls_name not in cat_name_to_id:
                continue

            for obj in objs:
                # Filtering: No outline or too few pixels
                if not obj['contours'] or obj['pixelCount'] <= 4:
                    continue

                valid_polys = [p for p in obj['contours'] if len(p) >= 6]
                if not valid_polys: continue

                ann = {
                    'segmentation': valid_polys,
                    'category_id': cat_name_to_id[cls_name],
                    'iscrowd': 0,
                    'area': obj['pixelCount'],
                    'bbox': xyxy_to_xywh(polys_to_boxes([valid_polys])).tolist()[0]
                }
                temp_annotations.append(ann)

    if not temp_annotations:
        return None

    return {
        'image_info': {'width': w, 'height': h, 'file_name': img_filename},
        'anns': temp_annotations
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Convert iSAID dataset')
    parser.add_argument('--outdir', default='./dataset/iSAID_patches', type=str)
    parser.add_argument('--datadir', default='./dataset/iSAID_patches', type=str)
    parser.add_argument('--set', default="train,val", type=str)
    parser.add_argument('--workers', default=8, type=int, help='Number of parallel workers')
    return parser.parse_args()


def convert_isaid_instance_only(data_dir, out_dir, split_set, workers):
    category_instancesonly = [
        'unlabeled', 'ship', 'storage_tank', 'baseball_diamond',
        'tennis_court', 'basketball_court', 'Ground_Track_Field',
        'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
        'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane', 'Harbor'
    ]

    categories = [{"id": i, "name": name} for i, name in enumerate(category_instancesonly) if name != 'unlabeled']
    cat_name_to_id = {c['name']: c['id'] for c in categories}

    sets = split_set.split(',')
    for data_set in sets:
        print(f'\n==== Processing {data_set} (Parallel Workers: {workers}) ====')
        ann_dir = os.path.join(data_dir, data_set, 'images')
        save_path = os.path.join(out_dir, data_set)
        os.makedirs(save_path, exist_ok=True)

        files = natsorted([f for f in os.listdir(ann_dir) if f.endswith('_instance_id_RGB.png')])

        final_images = []
        final_annotations = []
        img_id = 0
        ann_id = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_file = {executor.submit(process_single_image, f, ann_dir, cat_name_to_id): f for f in files}
            for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(files),
                               desc=f"Converting {data_set}"):
                result = future.result()

                if result is None:
                    continue

                img_info = result['image_info']
                img_info['id'] = img_id
                final_images.append(img_info)

                for ann in result['anns']:
                    ann['id'] = ann_id
                    ann['image_id'] = img_id
                    final_annotations.append(ann)
                    ann_id += 1

                img_id += 1

        output_data = {
            'info': {
                'description': 'iSAID Dataset converted for AI4RS',
                'url': 'https://github.com/wokaikaixinxin/ai4rs',
                'version': '1.0',
                'year': 2026,
                'contributor': 'User',
                'date_created': '2026/03/09'
            },
            'images': final_images,
            'categories': categories,
            'annotations': final_annotations
        }

        json_file = os.path.join(save_path, f'instancesonly_filtered_{data_set}.json')
        with open(json_file, 'w') as f:
            json.dump(output_data, f)

        print(f"Finished {data_set}:")
        print(f" - Original images: {len(files)}")
        print(f" - Filtered images (with labels): {len(final_images)}")
        print(f" - Total objects: {len(final_annotations)}")


if __name__ == '__main__':
    args = parse_args()
    convert_isaid_instance_only(args.datadir, args.outdir, args.set, args.workers)