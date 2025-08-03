import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def run_pytorch_benchmark(prompt, batch_size, seq_len, repeat):
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model.generate(**inputs, max_new_tokens=seq_len)
            torch.cuda.synchronize()

    latencies = []
    outputs = []
    with torch.no_grad():
        for _ in range(repeat):
            torch.cuda.synchronize()
            start = time.time()
            output_ids = model.generate(**inputs, max_new_tokens=seq_len)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append((end - start) * 1000)
            outputs.append(tokenizer.batch_decode(output_ids, skip_special_tokens=True))

    avg_lat = sum(latencies) / repeat
    shape = output_ids.shape
    return avg_lat, shape, outputs[0]  # 回傳第一個 decode
