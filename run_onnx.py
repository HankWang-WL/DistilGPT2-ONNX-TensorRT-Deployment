import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import time

def run_onnx_benchmark(prompt, batch_size, seq_len, repeat):
    onnx_model = "onnx/model.onnx"
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    session = ort.InferenceSession(onnx_model, providers=["CUDAExecutionProvider"])

    # 初始化 input
    inputs = tokenizer([prompt] * batch_size, return_tensors="np", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # ==== Warmup====
    for _ in range(10):
        input_feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        session.run(None, input_feed)
        
    latencies = []
    decodes = []
    vocab_size = 50257
    for _ in range(repeat):
        cur_input_ids = input_ids.copy()
        cur_attn_mask = attention_mask.copy()
        generated = []

        start = time.time()
        for _ in range(seq_len):
            input_feed = {
                "input_ids": cur_input_ids,
                "attention_mask": cur_attn_mask,
            }
            logits = session.run(None, input_feed)[0]  # [B, seq, V]
            # 取最後一個 token logits
            next_token_logits = logits[:, -1, :]  # [B, V]
            next_token = np.argmax(next_token_logits, axis=-1).astype(np.int32)  # [B]
            # append
            cur_input_ids = np.concatenate([cur_input_ids, next_token[:, None]], axis=1)
            cur_attn_mask = np.concatenate([cur_attn_mask, np.ones((batch_size, 1), dtype=np.int32)], axis=1)
        end = time.time()

        latencies.append((end - start) * 1000)
        # decode
        decodes.append(tokenizer.batch_decode(cur_input_ids, skip_special_tokens=True))
    avg_lat = sum(latencies) / repeat
    shape = cur_input_ids.shape
    return avg_lat, shape, decodes[0]
