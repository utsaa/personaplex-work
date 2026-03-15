import tensorrt as trt
import sys

def inspect_engine(engine_path):
    logger = trt.Logger(trt.Logger.INFO)
    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print(f"Failed to load engine from {engine_path}")
        return

    print(f"Engine: {engine_path}")
    print(f"Number of IO tensors: {engine.num_io_tensors}")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        print(f" - [{mode}] Name: {name}, Dtype: {dtype}, Shape: {shape}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_engine.py <engine_path>")
    else:
        inspect_engine(sys.argv[1])
