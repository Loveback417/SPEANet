
## TensorRT

### Installation

#### Install ai4rs

Please follow the [installation guide](https://github.com/wokaikaixinxin/ai4rs?tab=readme-ov-file#installation-%EF%B8%8F-) to install ai4rs.

#### Install onnx

```
pip install onnx onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnx-simplifier -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### Install tensorrt

(1) check CUDA version
```
python -c "import torch; print(torch.version.cuda)"
```
For example, `12.1`

(2) check python version
```
python --version
```
For example, `Python 3.10.19`

(3) download TensorRT CUDA x.x tar package from [NVIDIA](https://developer.nvidia.com/tensorrt), and extract it to the current directory

```
# For example, TensorRT-10.10.0.31.Linux.x86_64-gnu.cuda-12.9.tar.gz
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.10.0/tars/TensorRT-10.10.0.31.Linux.x86_64-gnu.cuda-12.9.tar.gz
tar -xvf TensorRT-10.10.0.31.Linux.x86_64-gnu.cuda-12.9.tar.gz
pip install TensorRT-10.10.0.31/python/tensorrt-10.10.0.31-cp310-none-linux_x86_64.whl
export TENSORRT_DIR=$(pwd)/TensorRT-10.10.0.31
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH
```
check
```
echo $LD_LIBRARY_PATH
echo $TENSORRT_DIR
# /root/TensorRT-10.10.0.31/lib:
# /root/TensorRT-10.10.0.31
```

### Export

#### Onnx

```
python projects/easydeploy/tools/export_onnx_rtdetr.py
```

#### Onnx -> Tensorrt, use trtexec
For example:
```
/root/TensorRT-10.10.0.31/bin/trtexec --onnx=/root/ai4rs/work_dirs/easydeploy/rtdetr/rtdetr_r50vd_8xb2-72e_coco_ad2bdcfe.onnx --saveEngine=/root/ai4rs/work_dirs/easydeploy/rtdetr/rtdetr_r50vd_8xb2-72e_coco_ad2bdcfe.engine --fp16
```
Note: use your own path! Note: use your own path! 

#### Visual
```
python projects/easydeploy/tools/image_demo_rtdetr.py
```

#### Lantency, use trtexec
```
/root/TensorRT-10.10.0.31/bin/trtexec --avgRuns=1000 --useSpinWait --loadEngine=/root/ai4rs/work_dirs/easydeploy/rtdetr/rtdetr_r50vd_8xb2-72e_coco_ad2bdcfe.engine
```


## Acknowledgement 🙏

[TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO)

[MMYOLO](https://github.com/open-mmlab/mmyolo)

[RTDETR](https://github.com/lyuwenyu/RT-DETR)