
Bilibili Deploy Tutorial: [![Bilibili](https://img.shields.io/badge/Deploy_Tutorial-fb7299?style=flat-square&logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV1VmwLzWExY/)

## TensorRT

### Installation

#### Install ai4rs

Please follow the [installation guide](https://github.com/wokaikaixinxin/ai4rs?tab=readme-ov-file#installation-%EF%B8%8F-) to install ai4rs.

#### Install onnx

```bash
pip install onnx onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnx-simplifier -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### Install tensorrt

(1) check CUDA version
```bash
python -c "import torch; print(torch.version.cuda)"
```
For example, `12.1`

(2) check python version
```bash
python --version
```
For example, `Python 3.10.20`

(3) download TensorRT CUDA x.x tar package from [NVIDIA](https://developer.nvidia.com/tensorrt), and extract it to the current directory

```bash
# For example, TensorRT-10.10.0.31.Linux.x86_64-gnu.cuda-12.9.tar.gz
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.10.0/tars/TensorRT-10.10.0.31.Linux.x86_64-gnu.cuda-12.9.tar.gz
tar -xvf TensorRT-10.10.0.31.Linux.x86_64-gnu.cuda-12.9.tar.gz
pip install TensorRT-10.10.0.31/python/tensorrt-10.10.0.31-cp310-none-linux_x86_64.whl
export TENSORRT_DIR=$(pwd)/TensorRT-10.10.0.31
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH
```

check

```bash
echo $LD_LIBRARY_PATH
echo $TENSORRT_DIR
# /root/TensorRT-10.10.0.31/lib:
# /root/TensorRT-10.10.0.31
```

### Export

#### Onnx

```
cd projects/easydeploy/tools/
python export_onnx_o2_rtdetr.py
```

#### Onnx -> Tensorrt, use trtexec

```
/root/TensorRT-10.10.0.31/bin/trtexec --onnx=/root/ai4rs/work_dirs/easydeploy/o2_rtdetr/epoch_72.onnx --saveEngine=/root/ai4rs/work_dirs/easydeploy/o2_rtdetr/epoch_72.engine --fp16
```

Note: use your own path! Note: use your own path! 


#### Lantency, use trtexec

```
/root/TensorRT-10.10.0.31/bin/trtexec --avgRuns=1000 --useSpinWait --loadEngine=/root/ai4rs/work_dirs/easydeploy/o2_rtdetr/epoch_72.engine
```

#### Visual
```
python image_demo_o2_rtdetr.py
```