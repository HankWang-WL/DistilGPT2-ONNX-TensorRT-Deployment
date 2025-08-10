import tensorrt as trt
import ctypes
import numpy as np
from transformers import AutoTokenizer
import time
import nvtx

def run_tensorrt_benchmark(prompt, batch_size, seq_len, repeat):
    CUDA_DLL_PATH = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\cudart64_12.dll"
    cuda = ctypes.CDLL(CUDA_DLL_PATH)
    # CUDA API signatures
    cuda.cudaMalloc.restype = int
    cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cuda.cudaMemcpy.restype = int
    cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cuda.cudaFree.restype = int
    cuda.cudaFree.argtypes = [ctypes.c_void_p]
    cuda.cudaDeviceSynchronize.restype = int
    cuda.cudaDeviceSynchronize.argtypes = []

    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2

    def check_cuda(status):
        if status != 0:
            raise RuntimeError(f"CUDA Error: {status}")

    # ---- TensorRT init ----
    engine_path = "distilgpt2_fp32.engine"
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # ---- Inputs ----
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer([prompt] * batch_size, return_tensors="np", padding=True)
    input_ids0 = inputs["input_ids"].astype(np.int32)
    attn_mask0 = inputs["attention_mask"].astype(np.int32)

    vocab_size = 50257
    latencies = []
    decodes = []

    # ---- O1: 預先配置 device buffer，一次到位、整段重用 ----
    init_len = int(input_ids0.shape[1])
    max_seq = init_len + seq_len
    bytes_i32 = np.dtype(np.int32).itemsize
    bytes_f32 = np.dtype(np.float32).itemsize

    d_input_ids = ctypes.c_void_p()
    d_attention_mask = ctypes.c_void_p()
    d_output = ctypes.c_void_p()
    check_cuda(cuda.cudaMalloc(ctypes.byref(d_input_ids),   batch_size * max_seq * bytes_i32))
    check_cuda(cuda.cudaMalloc(ctypes.byref(d_attention_mask), batch_size * max_seq * bytes_i32))
    check_cuda(cuda.cudaMalloc(ctypes.byref(d_output),      batch_size * max_seq * vocab_size * bytes_f32))

    # 只存最後一個 token 的 logits（host 端）
    last_logits = np.empty((batch_size, vocab_size), dtype=np.float32)

    # 小工具：只拷有效長度（H2D）
    def memcpy_htod_valid(d_ptr, h_arr, count_elems, elem_size):
        check_cuda(cuda.cudaMemcpy(
            d_ptr,
            ctypes.c_void_p(h_arr.ctypes.data),
            count_elems * elem_size,
            cudaMemcpyHostToDevice
        ))

    # 小工具：從 device 的 [B, cur_len, V] 只拷最後一個 timestep 的 [B, V] 回 host（D2H）
    def memcpy_dtoh_slice_last_token(d_output_ptr, cur_len):
        last_t = cur_len - 1
        for b in range(batch_size):
            offset_elems = (b * cur_len * vocab_size) + (last_t * vocab_size)
            src_ptr = ctypes.c_void_p(d_output_ptr.value + offset_elems * bytes_f32)
            dst_view = last_logits[b, :]  # [V]
            check_cuda(cuda.cudaMemcpy(
                ctypes.c_void_p(dst_view.ctypes.data),
                src_ptr,
                vocab_size * bytes_f32,
                cudaMemcpyDeviceToHost
            ))

    # ==== Warmup（沿用 O1 流程）====
    for _ in range(10):
        cur_input_ids = input_ids0.copy()
        cur_attn_mask = attn_mask0.copy()
        for _ in range(seq_len):
            cur_len = int(cur_input_ids.shape[1])
            context.set_binding_shape(0, (batch_size, cur_len))
            context.set_binding_shape(1, (batch_size, cur_len))

            # H2D：整條（O1 保持不變）
            memcpy_htod_valid(d_input_ids, cur_input_ids, batch_size * cur_len, bytes_i32)
            memcpy_htod_valid(d_attention_mask, cur_attn_mask, batch_size * cur_len, bytes_i32)

            # Compute
            context.execute_v2([int(d_input_ids.value), int(d_attention_mask.value), int(d_output.value)])
            cuda.cudaDeviceSynchronize()

            # D2H：只拿最後一 token 的 logits
            memcpy_dtoh_slice_last_token(d_output, cur_len)

            # greedy（用 last_logits）
            next_token = np.argmax(last_logits, axis=-1).astype(np.int32)
            cur_input_ids = np.concatenate([cur_input_ids, next_token[:, None]], axis=1)
            cur_attn_mask = np.concatenate([cur_attn_mask, np.ones((batch_size, 1), dtype=np.int32)], axis=1)

    # ==== benchmark (E2E；O1) ====
    with nvtx.annotate("TensorRT Benchmark (E2E O1)", color="red"):
        for _ in range(repeat):
            cur_input_ids = input_ids0.copy()
            cur_attn_mask = attn_mask0.copy()

            start = time.time()  # E2E：整段自回歸
            for _ in range(seq_len):
                cur_len = int(cur_input_ids.shape[1])
                context.set_binding_shape(0, (batch_size, cur_len))
                context.set_binding_shape(1, (batch_size, cur_len))

                # H2D：整條（O1 仍保持）
                memcpy_htod_valid(d_input_ids, cur_input_ids, batch_size * cur_len, bytes_i32)
                memcpy_htod_valid(d_attention_mask, cur_attn_mask, batch_size * cur_len, bytes_i32)

                # Compute
                context.execute_v2([int(d_input_ids.value), int(d_attention_mask.value), int(d_output.value)])
                cuda.cudaDeviceSynchronize()

                # D2H：只取最後一個 token 的 logits
                memcpy_dtoh_slice_last_token(d_output, cur_len)

                # greedy（CPU）
                next_token = np.argmax(last_logits, axis=-1).astype(np.int32)
                cur_input_ids = np.concatenate([cur_input_ids, next_token[:, None]], axis=1)
                cur_attn_mask = np.concatenate([cur_attn_mask, np.ones((batch_size, 1), dtype=np.int32)], axis=1)

            end = time.time()
            latencies.append((end - start) * 1000)
            decodes.append(tokenizer.batch_decode(cur_input_ids, skip_special_tokens=True))

    # 釋放（一次）
    check_cuda(cuda.cudaFree(d_input_ids))
    check_cuda(cuda.cudaFree(d_attention_mask))
    check_cuda(cuda.cudaFree(d_output))

    avg_lat = sum(latencies) / max(1, repeat)
    shape = cur_input_ids.shape
    return avg_lat, shape, decodes[0]
