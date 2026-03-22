import os
import tensorrt as trt
import torch
import numpy as np

class TRTModel:
    def __init__(self, engine_path, device="cuda:0"):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.device = torch.device(device)
        self.engine_path = engine_path
        
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from {engine_path}")
            
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=self.device)
        
        # Binding metadata
        self.input_names = []
        self.output_names = []
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def set_shapes(self, inputs):
        """Set dynamic input shapes for the execution context."""
        for name, tensor in inputs.items():
            if name in self.input_names:
                if not self.context.set_input_shape(name, tensor.shape):
                    # Usually means the shape is out of bounds (min/opt/max) defined during build
                    profile_index = self.context.active_optimization_profile
                    min_shape, opt_shape, max_shape = self.engine.get_tensor_profile_shape(name, profile_index)
                    raise ValueError(f"[TRT] Failed to set input shape for '{name}' to {tensor.shape}. "
                                     f"Engine range: {min_shape} .. {max_shape}")

    def run(self, inputs):
        """
        Runs inference on the provided inputs.
        
        Args:
            inputs (dict): Dictionary mapping input names to torch.Tensors (on device).
            
        Returns:
            dict: Dictionary mapping output names to torch.Tensors (on device).
        """
        # Set shapes for dynamic batch/frames
        self.set_shapes(inputs)
        
        # Prepare bindings and output tensors
        bindings = {}
        outputs = {}
        
        # Setup inputs
        for name, tensor in inputs.items():
            bindings[name] = tensor.data_ptr()
            
        # Setup outputs
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            # Allocation on the fly (TRT requires us to allocate buffers)
            # Use torch.empty for performance
            dtype = torch.float32 # Default
            # Map TRT dtype to torch dtype if possible
            trt_dtype = self.engine.get_tensor_dtype(name)
            if trt_dtype == trt.float32:
                dtype = torch.float32
            elif trt_dtype == trt.float16:
                dtype = torch.float16
                
            out_tensor = torch.empty(tuple(shape), dtype=dtype, device=self.device)
            outputs[name] = out_tensor
            bindings[name] = out_tensor.data_ptr()
            
        # Set tensor addresses
        for name, ptr in bindings.items():
            self.context.set_tensor_address(name, ptr)
            
        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        
        return outputs

    def __call__(self, *args, **kwargs):
        # Convenience method to map positional args if names are known, 
        # but dict inputs are safer for UNet.
        if len(args) > 0 and isinstance(args[0], dict):
            return self.run(args[0])
        return self.run(kwargs)
