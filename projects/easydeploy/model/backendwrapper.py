import warnings
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import onnxruntime

try:
    import tensorrt as trt
except Exception:
    trt = None
import torch

class TRTWrapper(torch.nn.Module):
    dtype_mapping = {}

    def __init__(self, weight: Union[str, Path],
                 device: Optional[torch.device]):
        super().__init__()
        weight = Path(weight) if isinstance(weight, str) else weight
        assert weight.exists() and weight.suffix in ('.engine', '.plan')
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f'cuda:{device}')
        self.weight = weight
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.__update_mapping()
        self.__init_engine()
        self.__init_bindings()

    def __update_mapping(self):
        self.dtype_mapping.update({
            trt.bool: torch.bool,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
            trt.float16: torch.float16,
            trt.float32: torch.float32
        })

    def __init_engine(self):
        logger = trt.Logger(trt.Logger.ERROR)
        self.log = partial(logger.log, trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, namespace='')
        self.logger = logger

        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        context = model.create_execution_context()

        names = [model.get_tensor_name(i) for i in range(model.num_io_tensors)]
        modes = [model.get_tensor_mode(name) for name in names]

        self.input_names = [n for n, m in zip(names, modes)
                            if m == trt.TensorIOMode.INPUT]
        self.output_names = [n for n, m in zip(names, modes)
                             if m == trt.TensorIOMode.OUTPUT]

        self.num_inputs = len(self.input_names)
        self.num_outputs = len(self.output_names)

        # dynamic?
        self.is_dynamic = any(
            -1 in model.get_tensor_shape(name)
            for name in self.input_names
        )

        self.model = model
        self.context = context
        self.bindings = {}

    def __init_bindings(self):
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape'))
        inputs_info = []
        outputs_info = []

        for name in self.input_names:
            dtype = self.dtype_mapping[self.model.get_tensor_dtype(name)]
            shape = tuple(self.model.get_tensor_shape(name))
            inputs_info.append(Binding(name, dtype, shape))

        for name in self.output_names:
            dtype = self.dtype_mapping[self.model.get_tensor_dtype(name)]
            shape = tuple(self.model.get_tensor_shape(name))
            outputs_info.append(Binding(name, dtype, shape))

        self.inputs_info = inputs_info
        self.outputs_info = outputs_info

        if not self.is_dynamic:
            self.output_tensor = [
                torch.empty(o.shape, dtype=o.dtype, device=self.device)
                for o in outputs_info
            ]

    def forward(self, *inputs):

        assert len(inputs) == self.num_inputs

        for name, tensor in zip(self.input_names, inputs):
            x = tensor.contiguous()
            self.bindings[name] = x.data_ptr()
            if self.is_dynamic:
                self.context.set_tensor_shape(name, tuple(x.shape))

        # output tensors
        outputs = []
        for idx, name in enumerate(self.output_names):

            if self.is_dynamic:
                shape = tuple(self.context.get_tensor_shape(name))
                out = torch.empty(shape,
                                  dtype=self.outputs_info[idx].dtype,
                                  device=self.device)
            else:
                out = self.output_tensor[idx]

            self.bindings[name] = out.data_ptr()
            outputs.append(out)

        # TRT10 requires set_tensor_address() for all tensors
        for name, ptr in self.bindings.items():
            self.context.set_tensor_address(name, ptr)

        # execute v3
        ok = self.context.execute_async_v3(self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

        self.stream.synchronize()
        return tuple(outputs)




class ORTWrapper(torch.nn.Module):

    def __init__(self, weight: Union[str, Path],
                 device: Optional[torch.device]):
        super().__init__()
        weight = Path(weight) if isinstance(weight, str) else weight
        assert weight.exists() and weight.suffix == '.onnx'

        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f'cuda:{device}')
        self.weight = weight
        self.device = device
        self.__init_session()
        self.__init_bindings()

    def __init_session(self):
        providers = ['CPUExecutionProvider']
        if 'cuda' in self.device.type:
            providers.insert(0, 'CUDAExecutionProvider')

        session = onnxruntime.InferenceSession(
            str(self.weight), providers=providers)
        self.session = session

    def __init_bindings(self):
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape'))
        inputs_info = []
        outputs_info = []
        self.is_dynamic = False
        for i, tensor in enumerate(self.session.get_inputs()):
            if any(not isinstance(i, int) for i in tensor.shape):
                self.is_dynamic = True
            inputs_info.append(
                Binding(tensor.name, tensor.type, tuple(tensor.shape)))

        for i, tensor in enumerate(self.session.get_outputs()):
            outputs_info.append(
                Binding(tensor.name, tensor.type, tuple(tensor.shape)))
        self.inputs_info = inputs_info
        self.outputs_info = outputs_info
        self.num_inputs = len(inputs_info)

    def forward(self, *inputs):

        assert len(inputs) == self.num_inputs

        contiguous_inputs: List[np.ndarray] = [
            i.contiguous().cpu().numpy() for i in inputs
        ]

        if not self.is_dynamic:
            # make sure input shape is right for static input shape
            for i in range(self.num_inputs):
                assert contiguous_inputs[i].shape == self.inputs_info[i].shape

        outputs = self.session.run([o.name for o in self.outputs_info], {
            j.name: contiguous_inputs[i]
            for i, j in enumerate(self.inputs_info)
        })

        return tuple(torch.from_numpy(o).to(self.device) for o in outputs)
