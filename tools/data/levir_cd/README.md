# Preparing LEVIR-CD Dataset

>[A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection](https://www.mdpi.com/2072-4292/12/10/1662)


## Download LEVIR-CD dataset

The LEVIR-CD dataset can be downloaded from [LEVIR-CD](https://justchenhao.github.io/LEVIR/) or 
[modelscope(魔塔)](https://modelscope.cn/datasets/OmniData/LEVIR-CD)


The data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── LEVIR-CD
│   │   ├── train
│   │   │   │   ├── A       # (445 png)
│   │   │   │   ├── B       # (445 png)
│   │   │   │   ├── label   # (445 png)
│   │   ├── val
│   │   │   │   ├── A       # (64 png)
│   │   │   │   ├── B       # (64 png)
│   │   │   │   ├── label   # (64 png)
│   │   ├── test
│   │   │   │   ├── A       # (128 png)
│   │   │   │   ├── B       # (128 png)
│   │   │   │   ├── label   # (128 png)
```


## Classes of LEVIR-CD

The 2 classes.

```
'classes':
(
    'unchanged', 'changed'
)
```


```bibtex
@Article{Chen2020,
AUTHOR = {Chen, Hao and Shi, Zhenwei},
TITLE = {A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection},
JOURNAL = {Remote Sensing},
VOLUME = {12},
YEAR = {2020},
NUMBER = {10},
ARTICLE-NUMBER = {1662},
URL = {https://www.mdpi.com/2072-4292/12/10/1662},
ISSN = {2072-4292},
DOI = {10.3390/rs12101662}
}
``` 