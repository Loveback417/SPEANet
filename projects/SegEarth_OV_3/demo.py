from PIL import Image
import requests
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import tempfile
from mmseg.structures import SegDataSample
from projects.SegEarth_OV_3.segearthov3_segmentor import SegEarthOV3Segmentation

img_path = 'resources/oem_koeln_50.tif'
url = 'https://gitee.com/IamDayu/seg-earth-ov-3/raw/master/resources/oem_koeln_50.tif'

directory = os.path.dirname(img_path)
if directory and not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Created directory: {directory}")
if not os.path.exists(img_path):
    print(f"File not found. Downloading from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        with open(img_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download file: {e}")
else:
    print(f"File found at {img_path}")

img = Image.open(img_path)
img_tensor = transforms.Compose([
    transforms.ToTensor(),
])(img).unsqueeze(0).to('cuda') # This variable is only a placeholder; the actual data is read within the model. (To be optimized)

data_sample = SegDataSample()
img_meta = {
    'img_path': img_path,
    'ori_shape': img.size[::-1] # H, W
}
data_sample.set_metainfo(img_meta)

bpt_path = 'projects/sam3/assets/bpe_simple_vocab_16e6.txt.gz'
checkpoint_path = 'work_dirs/segearthov3/sam3.pt'

name_list = ['background', 'bareland,barren', 'grass', 'road', 'car',
             'tree,forest', 'water,river', 'cropland', 'building,roof,house']

with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=True) as temp_file:
    content = '\n'.join(name_list)
    temp_file.write(content)
    temp_file.flush()
    temp_path = temp_file.name

    model = SegEarthOV3Segmentation(
        type='SegEarthOV3Segmentation',
        model_type='SAM3',
        classname_path=temp_path,
        prob_thd=0.1,
        confidence_threshold=0.1,
        slide_stride=512,
        slide_crop=512,
        bpt_path=bpt_path,
        checkpoint_path=checkpoint_path
    )

seg_pred = model.predict(img_tensor, data_samples=[data_sample])
seg_pred = seg_pred[0].pred_sem_seg.data.cpu().numpy().squeeze(0)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].axis('off')
ax[1].imshow(seg_pred, cmap='viridis')
ax[1].axis('off')
plt.tight_layout()
# plt.show()
plt.savefig('seg_pred.png', bbox_inches='tight')
print('Finished!')
