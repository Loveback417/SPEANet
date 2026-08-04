# MessDet (ICCV 2025)

> ICCV Link [Measuring the Impact of Rotation Equivariance on Aerial Object Detection](https://iccv.thecvf.com/virtual/2025/poster/145)

> ArXiv Link [Measuring the Impact of Rotation Equivariance on Aerial Object Detection](https://arxiv.org/abs/2507.09896)

> Official Code [Link](https://github.com/Nu1sance/MessDet)

<!-- [ALGORITHM] -->

## Abstract

Due to the arbitrary orientation of objects in aerial images, rotation equivariance is a critical property for aerial object detectors. However, recent studies on rotation-equivariant aerial object detection remain scarce. Most detectors rely on data augmentation to enable models to learn approximately rotation-equivariant features. A few detectors have constructed rotation-equivariant networks, but due to the breaking of strict rotation equivariance by typical downsampling processes, these networks only achieve approximately rotation-equivariant backbones. Whether strict rotation equivariance is necessary for aerial image object detection remains an open question. In this paper, we implement a strictly rotation-equivariant backbone and neck network with a more advanced network structure and compare it with approximately rotation-equivariant networks to quantitatively measure the impact of rotation equivariance on the performance of aerial image detectors. Additionally, leveraging the inherently grouped nature of rotation-equivariant features, we propose a multi-branch head network that reduces the parameter count while improving detection accuracy. Based on the aforementioned improvements, this study proposes the Multi-branch head rotation-equivariant single-stage Detector (MessDet), which achieves state-of-the-art performance on the challenging aerial image datasets DOTA-v1.0, DOTA-v1.5 and DIOR-R with an exceptionally low parameter count. The code will be made publicly available.

<div align=center>
<img src="https://github.com/Nu1sance/MessDet/blob/main/figs/fig1.jpg" height="360"/>
</div>

## Install

MessDet needs to install [e2cnn](https://github.com/QUVA-Lab/e2cnn) first.

```shell
pip install -e git+https://github.com/QUVA-Lab/e2cnn.git#egg=e2cnn
```

If you can't access GitHub:

```shell
pip install -e git+https://gitee.com/kiko888/e2cnn.git#egg=e2cnn
```

## Pretrained Backbones

Appr. RE-CSPNeXt: [ModelScope(魔塔)](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/messdet/recspnext_appr_reca-4d2fac8f.pth) / [GoogleDrive](https://drive.google.com/file/d/1J8EulR5CUljk8-Uc5k7LMNMD84le6dwm/view?usp=sharing) / [BaiduCloud](https://pan.baidu.com/s/162NgoL3VtPpCz9GQGjJA4A?pwd=72dj)

Str. RE-CSPNeXt: [ModelScope(魔塔)](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/messdet/recspnext_str_reca-1e07cda0.pth) / [GoogleDrive](https://drive.google.com/file/d/1RgXEbaMPtTheSTVgNh2DvQgydDknK4b4/view?usp=sharing) / [BaiduCloud](https://pan.baidu.com/s/1wiGYDRlHJe6DGdm8ZupFJw?pwd=ftdh)


## DOTA-v1.0

|  Model  |   Aug  | mAP  | AP50 | AP75 |lr schd | Params(M) | FLOPS(G) | batch size |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: |  :---: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Appr. MessDet |  RR   | 51.43 | 78.46  | 56.93 | 36e | 13.51  |   32.31  |    8=4gpu*<br>2img/gpu     |        [messdet_appr_4xb2_36e_dota.py](./configs/messdet_appr_4xb2_36e_dota.py)  |  [ModelScope(魔塔)](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/messdet/messdet_appr_4xb2-36e_dota-e091491e.pth) / [Google](https://drive.google.com/file/d/1w-Q2ZS1z7iRaTqkpvnaodX_-wDgyyYL5/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1lRBxl3-8ETvnxpqKgbvmOw?pwd=awn7) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/messdet/messdet_appr_4xb2_36e_dota_submission.zip) |

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.7846091868142074  
ap of each class: plane:0.8926923889873286, baseball-diamond:0.8455008592249684, bridge:0.562335287957187, ground-track-field:0.6601653984815651, small-vehicle:0.8140381198602986, large-vehicle:0.8535996426923605, ship:0.8887561479286522, tennis-court:0.9079355797955663, basketball-court:0.8825936985850944, storage-tank:0.8764667918277436, soccer-ball-field:0.6449882633193426, roundabout:0.6589501641336116, harbor:0.7818266683478938, swimming-pool:0.8210531542643942, helicopter:0.6782356368071035  
COCO style result:  
AP50: 0.7846091868142074  
AP75: 0.5692516878528501  
mAP: 0.5143371813246235

**Note**:  
1. We **do not retrain** this model. We **only evaluate** it.  
2. The checkpoint is taken from the official code [repository](https://github.com/Nu1sance/MessDet).  
3. The AP50 we obtained is 78.46, which is slightly higher than the 78.45 reported in the paper.  
4. The number of parameters we measured is 13.51M, whereas the paper reports 15.3M. Please verify this discrepancy yourself.  
5. The FLOPs were measured on an RTX 2080 Ti.  
6. It is claimed that the experiments were conducted on 4 GPUs with a batch size of 8. Please verify this yourself. 



|  Model  | Aug  | mAP  | AP50 | AP75 | lr schd |Params(M) | FLOPS(G) | batch size |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: |  :---: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Str. MessDet |  RR | 53.11 | 79.16 | 59.24 | 36e | 15.1  |  32.51  |    8=4gpu*<br>2img/gpu     |        [messdet_str_4xb2_36e_dota.py](./configs/messdet_str_4xb2_36e_dota.py)        |          [ModelScope(魔塔)](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/messdet/messdet_str_4xb2-36e_dota-a568e77f.pth) / [Google](https://drive.google.com/file/d/1KvoDLuII3lGTn9aOe1X73ef_u_pMYmsU/view?usp=sharing) / [Baidu](https://pan.baidu.com/s/1unU0fhGh99BtJw5rvA-tpg?pwd=6zn7) \|  [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/messdet/messdet_str_4xb2_36e_dota_submission.zip)                           |

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.7916369218972547  
ap of each class: plane:0.8806215783693647, baseball-diamond:0.8518090088454945, bridge:0.557651613440209, ground-track-field:0.7199134791154757, small-vehicle:0.8152537114438111, large-vehicle:0.858851727368285, ship:0.8896579604013924, tennis-court:0.9084536064313747, basketball-court:0.882531306950247, storage-tank:0.8865648805630454, soccer-ball-field:0.689761443239299, roundabout:0.6694239667590157, harbor:0.7883087907781687, swimming-pool:0.8262171151777561, helicopter:0.6495336395758808  
COCO style result:  
AP50: 0.7916369218972547  
AP75: 0.5924070293101686  
mAP: 0.5310806044862999

**Note**:  
1. We **do not retrain** this model. We **only evaluate** it.  
2. The checkpoint is taken from the official code [repository](https://github.com/Nu1sance/MessDet).  
3. The AP50 we obtained is 79.16, which is slightly higher than the 79.12 reported in the paper.  
4. The number of parameters we measured is 15.1M, whereas the paper reports 18.1M. Please verify this discrepancy yourself.  
5. The FLOPs were measured on an RTX 2080 Ti.  
6. It is claimed that the experiments were conducted on 4 GPUs with a batch size of 8. Please verify this yourself. 

**Train**:

```
bash tools/dist_train.sh projects/MessDet/configs/messdet_appr_4xb2_36e_dota.py 4

bash tools/dist_train.sh projects/MessDet/configs/messdet_str_4xb2_36e_dota.py 4
```

**Test**:

```
bash tools/dist_test.sh projects/MessDet/configs/messdet_appr_4xb2_36e_dota.py work_dirs/messdet_appr_4xb2-36e_dota-e091491e.pth 4

bash tools/dist_test.sh projects/MessDet/configs/messdet_str_4xb2_36e_dota.py work_dirs/messdet_str_4xb2-36e_dota-a568e77f.pth 4
```

**Get Params and FLOPS**:

```
python tools/analysis_tools/get_flops.py projects/MessDet/configs/messdet_appr_4xb2_36e_dota.py

python tools/analysis_tools/get_flops.py projects/MessDet/configs/messdet_str_4xb2_36e_dota.py 
```




## DOTA-v1.5

|  Model  |   Aug  | mAP  | AP50 | AP75 | lr schd | Params(M) | FLOPS(G) | batch size |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: |  :---: | :---: | :---: | :---: |  :---: |:-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Appr. MessDet |   RR   | 46.15  | 72.32  | 48.17 | 36e | 13.51  | 32.32  |    8=4gpu*<br>2img/gpu     |        [messdet_appr_4xb2_36e_dota15.py](./configs/messdet_appr_4xb2_36e_dota15.py) |   [ModelScope(魔塔)](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/messdet/messdet_appr_4xb2-36e_dota15-deb9875b.pth) / [Google](https://drive.google.com/file/d/1W92Xrg2y6xCVNeomSKQRVxMMnNNh_jSG/view?usp=sharin) /  [Baidu](https://pan.baidu.com/s/1L8b0rc-nMucoTi5GdgmUYA?pwd=b6ie) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/messdet/messdet_appr_4xb2_36e_dota15_submission.zip) |


This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.7231559102214252  
ap of each class: plane:0.8061488290599665, baseball-diamond:0.8495408968936475, bridge:0.5439527287048158, ground-track-field:0.6520534203883611, small-vehicle:0.7360202940363386, large-vehicle:0.8272498526602193, ship:0.897292306436622, tennis-court:0.9083598641718541, basketball-court:0.8241169126689202, storage-tank:0.8195205992939426, soccer-ball-field:0.5909444011084335, roundabout:0.728401342096033, harbor:0.7758810871856663, swimming-pool:0.7506142202432586, helicopter:0.6532120711333305, container-crane:0.20718573746139474  
COCO style result:  
AP50: 0.7231559102214252  
AP75: 0.4816565350711702  
mAP: 0.46151141221594927

**Note**:  
1. We **do not retrain** this model. We **only evaluate** it.  
2. The checkpoint is taken from the official code [repository](https://github.com/Nu1sance/MessDet).  
3. The AP50 we obtained is 72.31, which is slightly lower than the 72.38 reported in the paper.  
4. The number of parameters we measured is 13.51M, whereas the paper reports 15.3M. Please verify this discrepancy yourself.  
5. The FLOPs were measured on an RTX 2080 Ti.  
6. It is claimed that the experiments were conducted on 4 GPUs with a batch size of 8. Please verify this yourself. 

|  Model  |   Aug  | mAP  | AP50 | AP75 | lr schd | Params(M) | FLOPS(G) | batch size |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: |  :---: | :---: | :---: | :---: |  :---: |:-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Str. MessDet |  RR   | 46.81  | 73.13 | 49.46 | 36e |  15.1  |  32.52   |    8=4gpu*<br>2img/gpu     |        [messdet_str_4xb2_36e_dota15.py](./configs/messdet_str_4xb2_36e_dota15.py)   |  [ModelScope(魔塔)](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/messdet/messdet_str_4xb2-36e_dota15-3f485450.pth) \ [Google](https://drive.google.com/file/d/1pCTV8Aq-QIIJu0MdynGGsuxunD4u8oxc/view?usp=sharing) \ [Baidu](https://pan.baidu.com/s/1TaU46Ga8O5YRQZiyL8tMPA?pwd=ih46) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/messdet/messdet_str_4xb2_36e_dota15_submission.zip) |

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.731274576717053  
ap of each class: plane:0.8082916865072621, baseball-diamond:0.8626075084668452, bridge:0.5551091228857752, ground-track-field:0.6371745766633555, small-vehicle:0.7423922345847895, large-vehicle:0.8271649386212907, ship:0.8979739899778031, tennis-court:0.9085602631717731, basketball-court:0.8338612649881125, storage-tank:0.8231861062620187, soccer-ball-field:0.6396722038827681, roundabout:0.7125884076829585, harbor:0.7746282628747976, swimming-pool:0.7606953099103054, helicopter:0.6611043282186657, container-crane:0.2553830227743271  
COCO style result:  
AP50: 0.731274576717053  
AP75: 0.49463701442870067  
mAP: 0.4681070414106633

**Note**:  
1. We **do not retrain** this model. We **only evaluate** it.  
2. The checkpoint is taken from the official code [repository](https://github.com/Nu1sance/MessDet).  
3. The AP50 we obtained is 73.13, which is slightly lower than the 73.14 reported in the paper.  
4. The number of parameters we measured is 15.1M, whereas the paper reports 18.1M. Please verify this discrepancy yourself.  
5. The FLOPs were measured on an RTX 2080 Ti.  
6. It is claimed that the experiments were conducted on 4 GPUs with a batch size of 8. Please verify this yourself. 

**Train**:

```
bash tools/dist_train.sh projects/MessDet/configs/messdet_appr_4xb2_36e_dota15.py 4

bash tools/dist_train.sh projects/MessDet/configs/messdet_str_4xb2_36e_dota15.py 4
```

**Test**:

```
bash tools/dist_test.sh projects/MessDet/configs/messdet_appr_4xb2_36e_dota15.py work_dirs/messdet_appr_4xb2-36e_dota15-deb9875b.pth 4

bash tools/dist_test.sh projects/MessDet/configs/messdet_str_4xb2_36e_dota15.py work_dirs/messdet_str_4xb2-36e_dota15-3f485450.pth 4
```

**Get Params and FLOPS**:

```
python tools/analysis_tools/get_flops.py projects/MessDet/configs/messdet_appr_4xb2_36e_dota15.py

python tools/analysis_tools/get_flops.py projects/MessDet/configs/messdet_str_4xb2_36e_dota15.py
```

## Verify strict rotation equivariance

In addition, we provide a script to verify strict rotation equivariance, located at messdet/tools/check_rotation_equivariant.py. Users can run it to observe the equivariance error.

```
python projects/MessDet/messdet/check_rotation_equivariant.py
```

```
UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead. (Triggered internally at /opt/conda/conda-bld/pytorch_1702400410390/work/aten/src/ATen/native/IndexingUtils.h:27.)
  full_mask[mask] = norms.to(torch.uint8)
For 0 angle rotation, model's equivariance: YES
Equivariance Errors: 0.0
For 45 angle rotation, model's equivariance: YES
Equivariance Errors: 3.550877647473527e-11
For 90 angle rotation, model's equivariance: YES
Equivariance Errors: 8.164522520617823e-16
For 135 angle rotation, model's equivariance: YES
Equivariance Errors: 3.550889790537859e-11
For 180 angle rotation, model's equivariance: YES
Equivariance Errors: 9.361785062716056e-16
For 225 angle rotation, model's equivariance: YES
Equivariance Errors: 3.550876953584137e-11
For 270 angle rotation, model's equivariance: YES
Equivariance Errors: 1.2022751772009649e-15
For 315 angle rotation, model's equivariance: YES
Equivariance Errors: 3.550877994418222e-11
```

## Citation

```
@article{wu2025measuring,
  title={Measuring the Impact of Rotation Equivariance on Aerial Object Detection},
  author={Wu, Xiuyu and Wang, Xinhao and Zhu, Xiubin and Yang, Lan and Liu, Jiyuan and Hu, Xingchen},
  journal={arXiv preprint arXiv:2507.09896},
  year={2025}
}
```
