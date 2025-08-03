import tensorrt as trt
import ctypes
import numpy as np
from transformers import AutoTokenizer
import time
import nvtx

def run_tensorrt_benchmark(prompt, batch_size, seq_len, repeat):
    CUDA_DLL_PATH = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\cudart64_12.dll"
    cuda = ctypes.CDLL(CUDA_DLL_PATH)
    cuda.cudaMalloc.restype = int
    cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cuda.cudaMemcpy.restype = int
    cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cuda.cudaFree.restype = int
    cuda.cudaFree.argtypes = [ctypes.c_void_p]
    cuda.cudaDeviceSynchronize.restype = int
    cuda.cudaDeviceSynchronize.argtypes = []

    def check_cuda(status):
        if status != 0:
            raise RuntimeError(f"CUDA Error: {status}")

    engine_path = "distilgpt2_fp32.engine"
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer([prompt] * batch_size, return_tensors="np", padding=True)
    input_ids = inputs["input_ids"].astype(np.int32)
    attention_mask = inputs["attention_mask"].astype(np.int32)

    def malloc_buf(arr):
        ptr = ctypes.c_void_p()
        check_cuda(cuda.cudaMalloc(ctypes.byref(ptr), arr.nbytes))
        return ptr

    def memcpy_htod(d_ptr, h_arr):
        check_cuda(cuda.cudaMemcpy(
            d_ptr,
            ctypes.c_void_p(h_arr.ctypes.data),
            h_arr.nbytes,
            1
        ))

    def memcpy_dtoh(h_arr, d_ptr):
        check_cuda(cuda.cudaMemcpy(
            ctypes.c_void_p(h_arr.ctypes.data),
            d_ptr,
            h_arr.nbytes,
            2
        ))

    vocab_size = 50257
    latencies = []
    decodes = []
    # ==== Warmup====
    for _ in range(10):
        cur_input_ids = input_ids.copy()
        cur_attn_mask = attention_mask.copy()
        batch, seq = cur_input_ids.shape
        for _ in range(seq_len):
            context.set_binding_shape(0, cur_input_ids.shape)
            context.set_binding_shape(1, cur_attn_mask.shape)

            d_input_ids = malloc_buf(cur_input_ids)
            d_attention_mask = malloc_buf(cur_attn_mask)
            output_shape = (batch, cur_input_ids.shape[1], vocab_size)
            output = np.empty(output_shape, dtype=np.float32)
            d_output = malloc_buf(output)

            bindings = [int(d_input_ids.value), int(d_attention_mask.value), int(d_output.value)]

            memcpy_htod(d_input_ids, cur_input_ids)
            memcpy_htod(d_attention_mask, cur_attn_mask)

            context.execute_v2(bindings)
            cuda.cudaDeviceSynchronize()

            next_token_logits = output[:, -1, :]
            next_token = np.argmax(next_token_logits, axis=-1).astype(np.int32)
            cur_input_ids = np.concatenate([cur_input_ids, next_token[:, None]], axis=1)
            cur_attn_mask = np.concatenate([cur_attn_mask, np.ones((batch, 1), dtype=np.int32)], axis=1)

            cuda.cudaFree(d_input_ids)
            cuda.cudaFree(d_attention_mask)
            cuda.cudaFree(d_output)
    # ==== benchmark ====
    with nvtx.annotate("TensorRT Benchmark", color="red"):
        for _ in range(repeat):
            cur_input_ids = input_ids.copy()
            cur_attn_mask = attention_mask.copy()
            batch, seq = cur_input_ids.shape

            # allocate output
            for _ in range(seq_len):
                context.set_binding_shape(0, cur_input_ids.shape)
                context.set_binding_shape(1, cur_attn_mask.shape)

                d_input_ids = malloc_buf(cur_input_ids)
                d_attention_mask = malloc_buf(cur_attn_mask)
                output_shape = (batch, cur_input_ids.shape[1], vocab_size)
                output = np.empty(output_shape, dtype=np.float32)
                d_output = malloc_buf(output)

                bindings = [int(d_input_ids.value), int(d_attention_mask.value), int(d_output.value)]

                memcpy_htod(d_input_ids, cur_input_ids)
                memcpy_htod(d_attention_mask, cur_attn_mask)

                start = time.time()
                context.execute_v2(bindings)
                cuda.cudaDeviceSynchronize()
                end = time.time()

                memcpy_dtoh(output, d_output)

                # greedy
                next_token_logits = output[:, -1, :]
                next_token = np.argmax(next_token_logits, axis=-1).astype(np.int32)
                cur_input_ids = np.concatenate([cur_input_ids, next_token[:, None]], axis=1)
                cur_attn_mask = np.concatenate([cur_attn_mask, np.ones((batch, 1), dtype=np.int32)], axis=1)

                # free memory
                cuda.cudaFree(d_input_ids)
                cuda.cudaFree(d_attention_mask)
                cuda.cudaFree(d_output)

            latencies.append((end - start) * 1000)
            decodes.append(tokenizer.batch_decode(cur_input_ids, skip_special_tokens=True))

    avg_lat = sum(latencies) / repeat
    shape = cur_input_ids.shape
    return avg_lat, shape, decodes[0]
