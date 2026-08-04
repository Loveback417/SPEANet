import argparse
import ast
import sys
import logging

try:
    import tensorrt as trt
except Exception:
    trt = None
import tensorrt as trt
import torch


def build_engine(onnx, engine, img=(1,3,640,640), scales=None, fp16=True, device="cuda:0"):
    device = torch.device(device)
    torch.cuda.set_device(device.index or 0)

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    config  = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(onnx):
        for i in range(parser.num_errors):
            print("ONNX parse error:", parser.get_error(i))
        raise RuntimeError("ONNX parsing failed.")

    # ----------------- shapes -----------------
    if scales is None:
        print(f"Building STATIC engine with shape {img}")
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            profile.set_shape(inp.name, img, img, img)
        config.add_optimization_profile(profile)
    else:
        min_shape, opt_shape, max_shape = scales
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        print(f"Building DYNAMIC engine with min={min_shape}, opt={opt_shape}, max={max_shape}")

    # ----------------- precision -----------------
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # ----------------- build & save -----------------
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed.")
    with open(engine, "wb") as f:
        f.write(serialized)
    print("Saved TensorRT engine:", engine)


def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT engine (static or dynamic).")
    parser.add_argument("--checkpoint", "-c", type=str, help="ONNX model file path")
    parser.add_argument("--engine", "-e", type=str, help="Output TensorRT engine file")
    parser.add_argument("--img-size", nargs="+", type=int, default=[640,640],
                        help="Static input shape H W (or N C H W). Only used when --scales is not set")
    parser.add_argument("--scales", type=str, default=None,
                        help="Optional dynamic scales as Python literal, e.g. "
                             "'[(1,3,320,320),(1,3,640,640),(1,3,1280,1280)]'. Overrides --img-size")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 if supported")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (e.g., cuda:0)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    return args


def main(args):
    if len(args.img_size) == 1:
        args.img_size = (1, 3, args.img_size[0], args.img_size[0])
    elif len(args.img_size) == 2:
        args.img_size = (1, 3, args.img_size[0], args.img_size[1])
    elif len(args.img_size) == 4:
        args.img_size = tuple(args.img_size)
    else:
        raise ValueError("--img-size must have length 1,2 or 4")

    if args.scales:
        try:
            parsed = ast.literal_eval(args.scales)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
                args.scales = [tuple(int(x) for x in s) for s in parsed]
            else:
                raise ValueError("scales must be a 3-element sequence (min,opt,max)")
        except Exception as e:
            raise ValueError(f"Could not parse --scales: {e}")
    else:
        args.scales = None

    logger = logging.getLogger("EngineBuilder")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    try:
        if args.scales is None:
            logger.info(f"Building STATIC engine with img={args.img_size}")
        else:
            logger.info(f"Building DYNAMIC engine with scales={args.scales}")

        build_engine(
            onnx=args.checkpoint,
            engine=args.engine,
            img=args.img_size,
            scales=args.scales,
            fp16=args.fp16,
            device=args.device
        )
        logger.info("Engine build finished successfully.")

    except Exception as e:
        logger.exception(f"Engine build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    args.checkpoint = '../../../work_dirs/easydeploy/rtdetr/rtdetr_r50vd_8xb2-72e_coco_ad2bdcfe.onnx'
    args.engine = '../../../work_dirs/easydeploy/rtdetr/rtdetr_r50vd_8xb2-72e_coco_ad2bdcfe.engine'
    args.img_size = (640, 640)
    args.fp16 = True
    args.verbose = True
    main(args)

