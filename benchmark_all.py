from run_pytorch import run_pytorch_benchmark
from run_onnx import run_onnx_benchmark
from run_tensorrt import run_tensorrt_benchmark

prompt = "TensorRT is"
batch_size = 8
seq_len = 28
repeat = 1  # 建議生成測試設小一點

print("===== Pytorch Benchmark =====")
lat, shape, dec = run_pytorch_benchmark(prompt, batch_size, seq_len, repeat)
print(f"PyTorch latency: {lat:.2f} ms, output shape: {shape}")
print("Decoded:", dec)

print("\n===== ONNX Runtime Benchmark =====")
lat, shape, dec = run_onnx_benchmark(prompt, batch_size, seq_len, repeat)
print(f"ONNX latency: {lat:.2f} ms, output shape: {shape}")
print("Decoded:", dec)


print("\n===== TensorRT Benchmark =====")
lat, shape, dec = run_tensorrt_benchmark(prompt, batch_size, seq_len, repeat)
print(f"TensorRT latency: {lat:.2f} ms, output shape: {shape}")
print("Decoded:", dec)
