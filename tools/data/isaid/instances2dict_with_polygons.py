import os
import cv2
import numpy as np
from collections import namedtuple

Label = namedtuple('Label', ['name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color',
                             'm_color'])

labels = [
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0), 0),
    Label('ship', 1, 0, 'transport', 1, True, False, (0, 0, 63), 4128768),
    Label('storage_tank', 2, 1, 'transport', 1, True, False, (0, 63, 63), 4144896),
    Label('baseball_diamond', 3, 2, 'land', 2, True, False, (0, 63, 0), 16128),
    Label('tennis_court', 4, 3, 'land', 2, True, False, (0, 63, 127), 8339200),
    Label('basketball_court', 5, 4, 'land', 2, True, False, (0, 63, 191), 12533504),
    Label('Ground_Track_Field', 6, 5, 'land', 2, True, False, (0, 63, 255), 16727808),
    Label('Bridge', 7, 6, 'land', 2, True, False, (0, 127, 63), 4161280),
    Label('Large_Vehicle', 8, 7, 'transport', 1, True, False, (0, 127, 127), 8355584),
    Label('Small_Vehicle', 9, 8, 'transport', 1, True, False, (0, 0, 127), 8323072),
    Label('Helicopter', 10, 9, 'transport', 1, True, False, (0, 0, 191), 12517376),
    Label('Swimming_pool', 11, 10, 'land', 2, True, False, (0, 0, 255), 16711680),
    Label('Roundabout', 12, 11, 'land', 2, True, False, (0, 191, 127), 8371968),
    Label('Soccer_ball_field', 13, 12, 'land', 2, True, False, (0, 127, 191), 12549888),
    Label('plane', 14, 13, 'transport', 1, True, False, (0, 127, 255), 16744192),
    Label('Harbor', 15, 14, 'transport', 1, True, False, (0, 100, 155), 10183680),
]

m2label = {label.m_color: label for label in labels}
label2id = {label.name: label.id for label in labels}


class Instance(object):
    """Represents a single instance in an image"""

    def __init__(self, imgNp, imgNp_seg, instID):
        self.instID = int(instID)
        # Get the color value (category) corresponding to this instance in the semantic map
        # Quickly get the category of the first matching pixel using a mask
        mask = (imgNp == instID)
        if not np.any(mask):
            self.labelID = 0
            self.pixelCount = 0
        else:
            # Assume one instanceID corresponds to only one semantic label
            self.labelID = int(imgNp_seg[mask][0])
            self.pixelCount = int(mask.sum())

        self.medDist = -1
        self.distConf = 0.0

    def toDict(self):
        return {
            "instID": self.instID,
            "labelID": self.labelID,
            "pixelCount": self.pixelCount,
            "medDist": self.medDist,
            "distConf": self.distConf
        }


def findContours_compat(*args, **kwargs):
    res = cv2.findContours(*args, **kwargs)
    return res[-2], res[-1]  # Returns (contours, hierarchy)


def rgb_to_24bit(img):
    if img is None: return None
    img = img.astype(np.int64)
    return img[:, :, 0] + 256 * img[:, :, 1] + 65536 * img[:, :, 2]

def instances2dict_with_polygons(seg_imageFileList, ins_imageFileList, verbose=False):
    instanceDict = {}

    if not isinstance(seg_imageFileList, list):
        seg_imageFileList = [seg_imageFileList]
    if not isinstance(ins_imageFileList, list):
        ins_imageFileList = [ins_imageFileList]

    for img_seg_path, img_ins_path in zip(seg_imageFileList, ins_imageFileList):
        if verbose:
            print(f"Processing: {os.path.basename(img_ins_path)}")

        # Read images and convert to RGB
        img_ins = cv2.imread(img_ins_path)
        img_seg = cv2.imread(img_seg_path)

        if img_ins is None or img_seg is None:
            print(f"Warning: Could not read {img_ins_path} or {img_seg_path}")
            continue

        img_ins = cv2.cvtColor(img_ins, cv2.COLOR_BGR2RGB)
        img_seg = cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB)

        # Convert to 24-bit ID
        imgNp_ins = rgb_to_24bit(img_ins)
        imgNp_seg = rgb_to_24bit(img_seg)

        # Initialize result container
        current_img_instances = {label.name: [] for label in labels}

        # Get all unique Instance IDs
        unique_inst_ids = np.unique(imgNp_ins)

        for instID in unique_inst_ids:
            # iSAID convention: instID < 1000 is usually background or ignored region
            if instID < 1000:
                continue

            # Create instance object
            instanceObj = Instance(imgNp_ins, imgNp_seg, instID)

            # Check if the category exists in predefined labels
            if instanceObj.labelID not in m2label:
                continue

            label_info = m2label[instanceObj.labelID]
            instance_data = instanceObj.toDict()

            # If this category supports instance segmentation, extract polygons
            if label_info.hasInstances:
                mask = (imgNp_ins == instID).astype(np.uint8)
                contours, _ = findContours_compat(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Filter out polygons with too few points (at least 3 points, i.e., 6 values)
                polygons = [c.reshape(-1).tolist() for c in contours if len(c) >= 3]
                instance_data['contours'] = polygons

            current_img_instances[label_info.name].append(instance_data)

        imgKey = os.path.abspath(img_ins_path)
        instanceDict[imgKey] = current_img_instances

    return instanceDict