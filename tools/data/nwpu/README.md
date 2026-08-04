# NWPU VHR-10 (ISPRS 2014)

Download the NWPU VHR-10 dataset from [NWPU VHR-10](https://github.com/KyanChen/RSPrompter/tree/release/data/NWPU).


## Dataset preparation

Make sure that the final dataset must have this structure:
```
nwpu
├── imgs (650 jpg)
│    ├── 001.jpg
│    └── ...
│    └── 650.jpg
├── annotations
│    ├── NWPU_instances_train.json
│    ├── NWPU_instances_val.json
```



## Classes of NWPU VHR-10

The 12 classes.

```
'classes':
('airplane', 'ship', 'storage_tank', 'baseball_diamond',
'tennis_court', 'basketball_court', 'ground_track_field',
'harbor', 'bridge', 'vehicle')
```



## Description

The rapid development of remote sensing technology has facilitated us the acquisition of remote sensing images with higher and higher spatial resolution, but how to automatically understand the image contents is still a big challenge. In this paper, we develop a practical and rotation-invariant framework for multi-class geospatial object detection and geographic image classification based on collection of part detectors (COPD). The COPD is composed of a set of representative and discriminative part detectors, where each part detector is a linear support vector machine (SVM) classifier used for the detection of objects or recurring spatial patterns within a certain range of orientation. Specifically, when performing multi-class geospatial object detection, we learn a set of seed-based part detectors where each part detector corresponds to a particular viewpoint of an object class, so the collection of them provides a solution for rotation-invariant detection of multi-class objects. When performing geographic image classification, we utilize a large number of pre-trained part detectors to discovery distinctive visual parts from images and use them as attributes to represent the images.

[Paper link](https://www.sciencedirect.com/science/article/pii/S0924271614002524)



```bibtex
@article{cheng2014multi,
  title={Multi-class geospatial object detection and geographic image classification based on collection of part detectors},
  author={Cheng, Gong and Han, Junwei and Zhou, Peicheng and Guo, Lei},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={98},
  pages={119--132},
  year={2014},
  publisher={Elsevier}
}
``` 