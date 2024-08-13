import argparse
import json
import time

import os
from pathlib import Path

import torch
import tensorrt as trt
from depth_anything import DepthAnything


def export(
    model_path: str,  
    save_path: str,
    input_size: int,
    onnx: bool = True,
):  
    """
    model_path: str -> Path to the PyTorch model(local / hub)
    save_path: str -> Directory to save the model
    input_size: int -> Width and height of the input image(e.g. 308, 364, 406, 518)
    onnx: bool -> Export the model to ONNX format
    """
    weights_path = Path(model_path) / 'pytorch_model.bin'
    cfg_path = Path(model_path) / 'config.json'
    
    os.makedirs(save_path, exist_ok=True)
    # Config path
    with open(cfg_path) as f:
        cfg = json.load(f)
        assert cfg
        
    # 1. Load the model
    # need to install: pip install huggingface_hub
    # model = DepthAnything.from_pretrained(weights_path).to('cpu').eval()
    
    weights = torch.load(weights_path)
    model = DepthAnything(cfg).to('cpu').eval()
    model.load_state_dict(weights)
    
    # create a dummy input for export onnx
    dummy_input = torch.ones((3, input_size, input_size)).unsqueeze(0)
    _ = model(dummy_input)
    onnx_path = Path(save_path) / f"depth_anything_vits14_{input_size}.onnx"
    
    # 2. Export the PyTorch model to ONNX format
    if onnx:
        torch.onnx.export(
            model,
            dummy_input, 
            onnx_path, 
            opset_version=11, 
            input_names=["input"], 
            output_names=["output"], 
        )
        print(f"Model exported to {onnx_path}", onnx_path)
        print("Saving the model to ONNX format...")
        time.sleep(2)
    
    # 3. Convert the ONNX model to TensorRT engine
    
    # create trt logger
    logger = trt.Logger(trt.Logger.VERBOSE)
    # create builder
    builder = trt.Builder(logger)
    # create trt empty network
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # create onnx model parser
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, "rb") as model:
        # parse onnx model and write to trt network
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('Failed to parse the ONNX model.')
    
    # set up the builder config
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16) # FP16 inference
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2 GB
    # build trt serialized engine
    serialized_engine = builder.build_serialized_network(network, config)
    
    with open(onnx_path.with_suffix(".trt"), "wb") as f:
        f.write(serialized_engine)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export script for Depth-Anything-for-Jetson-Orin-Python')
    parser.add_argument("--weights_path", type=str, default="./ckpt")
    parser.add_argument("--save_path", type=str, default="exported_models")
    parser.add_argument("--input_size", type=int, default=364)
    args = parser.parse_args()
    
    export(
        model_path=args.weights_path,
        save_path=args.save_path,
        input_size=args.input_size,
        onnx=True,
    )
