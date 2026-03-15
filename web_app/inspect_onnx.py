import onnx
import sys

def inspect_onnx(onnx_path):
    model = onnx.load(onnx_path)
    print(f"ONNX Model: {onnx_path}")
    print("Inputs:")
    for input in model.graph.input:
        print(f" - {input.name}")
    print("Outputs:")
    for output in model.graph.output:
        print(f" - {output.name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_onnx.py <onnx_path>")
    else:
        inspect_onnx(sys.argv[1])
