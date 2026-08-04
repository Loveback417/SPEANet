# IDA-SiamNet  Interactive- and Dynamic-Aware Siamese Network for Building Change Detection (TGRS 2024)


> TGRS [Link](https://ieeexplore.ieee.org/document/10551864)

> Official Code [Link](https://github.com/SUPERMAN123000/IDA-SiamNet)



## Abstract
Building change detection (BCD) is a critical task in remote sensing which aims to identify the building changes within the same geographical area over time. The complexity of BCD is heightened when utilizing very high-resolution (VHR) remote sensing images, leading to two primary challenges: distinguishing between building and nonbuilding changes and accommodating the diverse range of building shapes and sizes. The existing mainstream methods neglect interactions
between encoders, thereby compromising the ability to recognize building and nonbuilding changes. Additionally, most BCD methods overlook feature alignment and fusion which hinders the precise extraction of buildings with varying shapes and sizes. To address these limitations, we propose an interactiveand dynamic-aware Siamese network (IDA-SiamNet) for BCD. Our method comprises the spatial exchange feature interaction (SEFI) module, the channel exchange feature interaction (CEFI) module, and the dynamic-deformable dual-alignment fusion (D3AF) module. The SEFI and CEFI modules play a pivotal role in facilitating mutual information exchange between Siamese encoders, enhancing discrimination between building and nonbuilding changes. Furthermore, the D3AF module dynamically aggregates multiple parallel convolutional kernels to improve feature alignment and fusion for accurate building outline extraction. D3AF adapts its receptive field (RF) based on object size and covers diverse building shapes without introducing excessive background information. Experimental evaluations on three widely used BCD datasets, learning, vision, and remote sensing change detection (LEVIR-CD), satellite side-looking (S2Looking), and WHU BCD (WHU-CD), demonstrate the superior performance of our proposed method over state-ofthe-art alternatives


```bibtex
@ARTICLE{10551864,
  author={Li, Yun-Cheng and Lei, Sen and Liu, Nanqing and Li, Heng-Chao and Du, Qian},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={IDA-SiamNet: Interactive- and Dynamic-Aware Siamese Network for Building Change Detection}, 
  year={2024},
  volume={62},
  number={},
  pages={1-13},
  keywords={Buildings;Feature extraction;Shape;Transformers;Remote sensing;Decoding;Architecture;Building change detection (BCD);feature alignment;feature interaction;remote sensing image;Siamese network},
  doi={10.1109/TGRS.2024.3410977}}
```

## Results and models

### LEVIR-CD


| Method | Backbone | Crop Size | Lr schd | Precision | Recall | F1-Score |  IoU   |                            config                            | download |
| :----: | :------: | :-------: | :-----: |  :-------: | :----: | :------: | :---: |  :----------------------------------------------------------: | :------: |
|  idasiamnet   |   r18     |  512x512  |  40000  |  93.17  | 90.54  | 91.84   | 84.91  | [config](./configs/idasiamnet_ex_r18_512x512_40k_levircd.py) | [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/idasiamnet/idasiam_r18_levir/best_mIoU_iter_40000.pth) |
|  idasiamnet   |   mit-b1  |  512x512  |  36000  |  93.49  | 90.74  | 92.10   | 85.35  | [config](./configs/idasiamnet_ex_mit-b1_512x512_40k_levircd.py) | [model](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/idasiamnet/idasiam_mit_levir/best_mIoU_iter_36000.pth) |

- Checkpoints are from official repo.
- All metrics are based on the category "change".
- All scores are computed on the test set.


train:

```bash
python tools/train.py projects/idasiamnet/configs/idasiamnet_ex_r18_512x512_40k_levircd.py
python tools/train.py projects/idasiamnet/configs/idasiamnet_ex_mit-b1_512x512_40k_levircd.py
```

test:

```bash
python tools/test.py projects/idasiamnet/configs/idasiamnet_ex_r18_512x512_40k_levircd.py your_path/best_mIoU_iter_40000.pth
python tools/test.py projects/idasiamnet/configs/idasiamnet_ex_mit-b1_512x512_40k_levircd.py your_path/best_mIoU_iter_36000.pth
```