import cv2
import os
from natsort import natsorted
from glob import glob
from shutil import copyfile
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Optimized Image Splitting')
    parser.add_argument('--src', default='./data/iSAID', type=str, help='path for the original dataset')
    parser.add_argument('--tar', default='./data/split_isaid', type=str, help='path for saving the new dataset')
    parser.add_argument('--image_sub_folder', default='images', type=str, help='subfolder name')
    parser.add_argument('--set', default="train,val", type=str, help='evaluation mode')
    parser.add_argument('--patch_width', default=800, type=int, help='Width of the cropped image patch')
    parser.add_argument('--patch_height', default=800, type=int, help='Height of the cropped image patch')
    parser.add_argument('--overlap_area', default=200, type=int, help='Overlap area')
    parser.add_argument('--workers', default=8, type=int, help='Number of parallel workers')
    return parser.parse_args()


def process_single_image(file_name, src_path, tar_path, extras, patch_H, patch_W, overlap):
    """
    Core function to process a single image and all its corresponding masks
    """
    for extra in extras:
        filename = file_name + extra + '.png'
        full_filename = os.path.join(src_path, filename)

        img = cv2.imread(full_filename)
        if img is None:
            continue

        img_H, img_W, _ = img.shape

        # If image size is smaller than patch size, copy directly
        if img_H <= patch_H or img_W <= patch_W:
            copyfile(full_filename, os.path.join(tar_path, filename))
            continue

        # Calculate patch coordinates
        y_offsets = list(range(0, img_H - patch_H, patch_H - overlap))
        y_offsets.append(img_H - patch_H)

        x_offsets = list(range(0, img_W - patch_W, patch_W - overlap))
        x_offsets.append(img_W - patch_W)

        for y_str in sorted(list(set(y_offsets))):
            y_end = y_str + patch_H
            for x_str in sorted(list(set(x_offsets))):
                x_end = x_str + patch_W
                patch = img[y_str:y_end, x_str:x_end, :]

                patch_name = f"{file_name}_{y_str}_{y_end}_{x_str}_{x_end}{extra}.png"
                save_path = os.path.join(tar_path, patch_name)
                cv2.imwrite(save_path, patch)

    return file_name


def main(args):
    src = args.src
    tar = args.tar
    modes = args.set.split(',')
    patch_H, patch_W = args.patch_height, args.patch_width
    overlap = args.overlap_area

    print(f"========== Optimized Image Splitting Started (Parallel Workers: {args.workers}) ==========")

    for mode in modes:
        if mode in ['train', 'val']:
            extras = ['', '_instance_color_RGB', '_instance_id_RGB']
        else:
            extras = ['']

        src_path = os.path.join(src, mode, args.image_sub_folder)
        tar_path = os.path.join(tar, mode, args.image_sub_folder)
        os.makedirs(tar_path, exist_ok=True)

        all_files = glob(os.path.join(src_path, "*.png"))
        files = [os.path.basename(f).split('.')[0] for f in all_files if '_' not in os.path.basename(f)]
        files = natsorted(files)

        print(f"\nProcessing {mode} split, total {len(files)} original images:")

        worker_func = partial(
            process_single_image,
            src_path=src_path,
            tar_path=tar_path,
            extras=extras,
            patch_H=patch_H,
            patch_W=patch_W,
            overlap=overlap
        )

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(tqdm(executor.map(worker_func, files), total=len(files), desc=f"Progress", unit="img"))

    print("\n========== All Tasks Completed ==========")


if __name__ == '__main__':
    args = parse_args()
    main(args)