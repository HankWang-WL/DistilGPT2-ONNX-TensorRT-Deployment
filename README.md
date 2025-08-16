# DistilGPT2-ONNX-TensorRT-Deployment

> This README focuses on **end-to-end (E2E)** latency. It documents how I used Nsight/logs to locate bottlenecks, implemented a **minimal O1 optimization**, and why under this setup **TensorRT (E2E)** is still slower than **PyTorch (E2E)**.

---

## Project Overview

This project demonstrates exporting Hugging Face **DistilGPT2** to ONNX and running it with **NVIDIA TensorRT** on Windows (GTX 1060), with rigorous performance analysis.

* **Reproducible**: clear env, commands, and settings.
* **E2E metrics**: compare *real* end‑to‑end latency, not kernel-only time.
* **Debug → Verify → Tradeoffs**: Nsight Systems + ONNX verbose logs for evidence; apply targeted optimization (O1) and openly describe limitations.

> **Bottom line:** We measure true end-to-end latency. A minimal O1 (pre-alloc + last-step logits only) cuts E2E by **30–43%**, but PyTorch `generate` remains faster **because it uses a KV cache (O(L))** while our TRT path is **O(L²)**. **To close the gap, add a KV cache**



---

## Inference Pipeline Overview

```bash
PyTorch (native inference)
      ↓
Export to ONNX
      ↓
ONNX Runtime (ONNX inference)
      ↓
Build TensorRT Engine
      ↓
TensorRT (inference)
      ↓
Profiling & Debugging (Nsight, Verbose Logs)
```

* **PyTorch**: model dev & baseline.
* **ONNX**: exchange format.
* **ONNX Runtime**: general purpose engine (CPU/GPU).
* **TensorRT**: NVIDIA GPU‑specific; manual memory binding & data movement design.
* **Nsight Systems**: E2E timeline.
* **ONNX Verbose Logs**: op/device placement & inserted Memcpy nodes.

---

## Directory Structure

```
DistilGPT2-ONNX-TensorRT-Deployment/
├── onnx/                  # Exported ONNX models
├── distilgpt2_fp32.engine # TensorRT engine (FP32)
├── benchmark_all.py       # Run all benchmarks & aggregate
├── run_pytorch.py         # PyTorch baseline
├── run_onnx.py            # ONNX Runtime baseline
├── run_tensorrt.py        # TensorRT baseline 
├── run_tensorrt_O1.py     # TensorRT O1 (E2E: prealloc + D2H last-step only)
├── images/
│   ├── nsight_onnx_memcpy.PNG
│   ├── nsight_pytorch_profile.PNG
│   ├── tensorrt_pipeline_latency.PNG
│   └── tensorrtO1_pipeline_latency.PNG
├── nsight_reports/
│   └── 3benchmark1round.nsys-rep
└── README.md
```

---

## Environment & Version Details

* **OS**: Windows 10 (64‑bit)
* **GPU**: NVIDIA GeForce GTX 1060 3GB (Pascal, CC 6.1)
* **Python**: 3.10
* **PyTorch**: 2.1.0+cu121
* **Transformers**: 4.40.0
* **ONNX**: 1.14.1
* **ONNX Runtime**: 1.14.1 (CUDA)
* **TensorRT**: 8.6.1.6 (FP32 only on Pascal)
* **CUDA**: 12.x

*Check version info in your own environment with:*

```
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import transformers; print(transformers.__version__)"
python -c "import onnx; print(onnx.__version__); import onnxruntime; print(onnxruntime.__version__)"
trtexec --version
nvcc --version
nvidia-smi
```

**Note:**

* GTX 1060 (Pascal) is only supported up to TensorRT 8.6.x and CUDA 12.x, and *FP32 only*.
* ONNX Runtime must be installed with CUDA support for GPU acceleration.

---

## Exporting to ONNX

Export the DistilGPT2 PyTorch model to ONNX using Hugging Face Transformers utility:

```bash
python -m transformers.onnx --model=distilgpt2 onnx/model.onnx
```

* This command downloads pretrained distilgpt2, exports to `onnx/model.onnx`.
* This command exports the model with dynamic axes enabled by default (both batch size and sequence length are dynamic), allowing flexible input shapes for deployment with ONNX Runtime or TensorRT.
* If using `torch.onnx.export` directly, be sure to set
  `dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}, "attention_mask": {0: "batch_size", 1: "seq_len"}}`.

---

## Building the TensorRT Engine

With ONNX ready, build the TensorRT engine for optimized inference.

**Command used:**

```bash
trtexec --onnx=onnx/model.onnx --saveEngine=distilgpt2_fp32.engine \
  --minShapes=input_ids:1x1,attention_mask:1x1 \
  --optShapes=input_ids:8x16,attention_mask:8x16 \
  --maxShapes=input_ids:16x32,attention_mask:16x32
```

* **FP32** engine is built (only supported precision for GTX 1060).
* min/opt/max shapes specify the supported dynamic input profile:
  * **Batch size:** 1–16
  * **Sequence length:** 1–32 (fully dynamic; both `input_ids` and `attention_mask` must always have identical shapes)
* The engine accepts inputs **only** within these batch and sequence length ranges; providing a shape outside this profile will trigger runtime errors.

**Tip:**
* For autoregressive decoding, always ensure the dynamic shape profile covers the full prompt + generated tokens length. Without this, TensorRT will crash when the sequence grows beyond the profile limit.
* If you need longer generations, set a larger `--maxShapes` (e.g., `16x64`).

---

## Measurement Methodology & Fairness

**E2E definition**: One-time **H2D** (Host→Device) for the initial prompt → per-step **Compute** → per-step **D2H** (Device→Host) for the needed result; the whole decode loop is timed.

**Why this is not an apples‑to‑apples comparison**:

1. **PyTorch `generate` uses KV cache** (default `use_cache=True`) → per‑step **O(L)**; tensors stay on GPU; **almost no per‑step memcpy**.
2. **ONNX/TensorRT path here does not use KV cache** → per‑step **O(L²)** recompute; more H2D/D2H overall.
3. **Sync points**: TRT/ONNX Python loop calls `cudaDeviceSynchronize()` per step, making overlap harder; PyTorch’s internal loop is tighter.

### Why KV cache ⇒ per-step O(L)

**Context (step t, current length = t):**
- **With KV cache** – past **K/V** are stored on GPU. At step *t* we compute a new **Q_t**, dot it with cached **K[1..t-1]**, then weight cached **V[1..t-1]**.  
  ➜ **Per-step cost ≈ O(t)** (linear).

- **Without KV cache** – rebuild full attention for all *t* tokens every step (`t × t` attention).  
  ➜ **Per-step cost ≈ O(t²)** (quadratic).

**Complexity summary**

| Case              | Per-step cost | Total cost up to length L |
|-------------------|---------------|---------------------------|
| With KV cache     | **O(L)**      | **O(L²)**                 |
| Without KV cache  | **O(L²)**     | **O(L³)**                 |

**KV cache:** lets each new token only attend to history once (linear), instead of rebuilding the whole `t×t` attention every step (quadratic).




### Nsight Systems (overview)

**PyTorch** — decode loop stays mostly on GPU (few transfers).  
![PyTorch: GPU Execution Profile](images/nsight_pytorch_profile.PNG)

**ONNX** — many long `cudaMemcpyAsync` (red) → dominant E2E cost.  
![ONNX: Excessive cudaMemcpyAsync](images/nsight_onnx_memcpy.PNG)

**TensorRT** — green (host waits): including orange (kernels) + red (copies). **E2E includes kernels and copies**.  
![TensorRT: Pipeline Latency](images/tensorrt_pipeline_latency.PNG)

### GPU Timeline Color Legend & Latency Definitions

**How to read Nsight timelines:**
- **Orange** = `ExecutionContext::execute` (GPU kernels; device-side compute)
- **Red** = `cudaMemcpy` （Host→Device（H2D）/ Device→Host（D2H） transfers）
- **Green** = host/device sync (e.g., `cudaStreamSynchronize`), i.e., CPU waiting; it overlaps with device work and is **not additional GPU time** to be added again.

**Two common latency definitions:**
1) **Kernel-only** — measure just the GPU compute (e.g., `execute_v2` + a sync). Useful for kernel tuning, but excludes I/O.
2) **End-to-end (E2E)** — includes H2D → compute → D2H and any required synchronization. This best reflects real deployment behavior and is what this README reports as the primary metric.


---

## Debugging Storyline (how I found and fixed the metrics)

1. **Initial observation**: TensorRT looked fast under **kernel-only** timing.
2. **Switch to E2E**: I timed **H2D → compute → D2H** with NVTX ranges. **Nsight** revealed large memcpy bars and host waits that kernel-only had skipped.
3. **ONNX logs → hypothesis**: **ONNX verbose logs** showed CPU fallbacks and inserted Memcpy nodes. That explained ONNX’s E2E lag and suggested a direction: **transfers and per-step allocations are the real pain**. *(No TensorRT verbose was used.)*
4. **Apply the idea on TensorRT (O1)**: After moving to TRT (no CPU fallbacks), Nsight still showed transfers dominating. I implemented **O1**—**pre-allocate** device buffers and **copy back only the last-step logits**—to attack the transfer/alloc overhead directly.
5. **Results & correctness**: E2E dropped by **30–43%** and Nsight showed shorter memcpy bars. Decoded outputs matched PyTorch **token-by-token**. The main remaining gap is the **missing KV cache** (per-step O(L) vs O(L²)).

> Chain of evidence: kernel-only looked great → switch to E2E → ONNX logs expose CPU fallback/Memcpy → hypothesize transfer/alloc bottleneck → TRT O1 fix → measurable E2E win; remaining gap = KV cache.

---


## Benchmark Results (E2E)

### Fixed seq\_len = 12, varying batch

| batch | seq\_len | PyTorch (ms) | ONNX (ms) | TensorRT E2E (ms) | **TensorRT E2E O1 (ms)** |
| ----: | -------: | -----------: | --------: | ----------------: | -----------------------: |
|     1 |       12 |        56.37 |     34.36 |             66.61 |       **57.86** (↓13.1%) |
|     8 |       12 |        58.81 |     82.46 |            105.13 |       **73.24** (↓30.3%) |
|    16 |       12 |        60.45 |    131.51 |            170.67 |      **102.03** (↓40.2%) |

### Fixed batch = 8, varying seq\_len

| batch | seq\_len | PyTorch (ms) | ONNX (ms) | TensorRT E2E (ms) | **TensorRT E2E O1 (ms)** |
| ----: | -------: | -----------: | --------: | ----------------: | -----------------------: |
|     8 |        8 |        31.57 |     31.27 |             50.02 |       **34.22** (↓31.6%) |
|     8 |       16 |        88.35 |    131.67 |            170.80 |      **110.50** (↓35.3%) |
|     8 |       32 |       205.31 |    474.65 |            628.14 |      **355.64** (↓43.4%) |

## Accuracy Verification

To validate functional correctness, I compared decoded outputs from the TensorRT path against the original PyTorch model across multiple prompts, batches, and sequence lengths. For all tested settings, the decoded sequences **matched exactly** token-by-token. This confirms the TensorRT deployment preserves model behavior relative to the PyTorch baseline.


**Interpretation**

* **O1 optimization** (minimal changes) yields a **30–43% reduction in E2E latency**:

  * **Change A:** Pre‑allocate and reuse device buffers (remove per‑step malloc/free).
  * **Change B:** Transfer back only the last‑step logits `[B, vocab_size]` instead of the full `[B, cur_len, vocab_size]`.
* Even after O1, TensorRT remains slower than PyTorch on E2E because the current path lacks a **KV cache**: each step recomputes the entire sequence (O(L²)), and I/O still dominates.

> At first I only measured `execute_v2` (kernel‑window), which includes kernel execution plus host synchronization but excludes H2D/D2H transfers and malloc/free. Later I switched to reporting **true E2E latency** (H2D → kernels → D2H → sync), which reflects real deployment cost.

### Sanity Check: exe\_v2 vs E2E (TensorRT)

| batch, seq | TRT exe\_v2 (ms) | TRT E2E (ms) | TRT E2E O1 (ms) | E2E / exe\_v2 (×) | O1 drop |
| ---------- | ---------------: | -----------: | --------------: | ----------------: | ------: |
| b=1,  L=12 |            56.58 |        66.61 |           57.86 |             1.18× |   13.1% |
| b=8,  L=12 |            66.16 |       105.13 |           73.24 |             1.59× |   30.3% |
| b=16, L=12 |            91.85 |       170.67 |          102.03 |             1.86× |   40.2% |
| b=8,  L=8  |            33.43 |        50.02 |           34.22 |             1.50× |   31.6% |
| b=8,  L=16 |           101.30 |       170.80 |          110.50 |             1.69× |   35.3% |
| b=8,  L=32 |           338.33 |       628.14 |          355.64 |             1.86× |   43.4% |

> **Read:** `exe_v2` (kernel‑window) = kernel execution + host sync, without H2D/D2H or malloc/free. **E2E** = full pipeline cost including transfers and sync. **O1** reduces E2E by 30–43%, but without KV cache the O(L²) recomputation and transfers remain the bottleneck.


---

## Why TensorRT (E2E) < PyTorch (E2E) here

1. **Different algorithmic path**: PyTorch `generate` uses **KV cache** (O(L)) with an internal GPU loop; the ONNX/TRT path here has **no cache** (O(L²)).
2. **I/O and sync**: TRT/ONNX perform per‑step H2D/D2H and use sync APIs; PyTorch has near‑zero per‑step I/O.
3. **Hardware**: GTX 1060 (Pascal, FP32‑only) has weaker copy/compute overlap; the gap is amplified under this design.


---

## Profiling & Debugging Analysis (evidence)

### GPU timeline: how to read it (quick legend)

* **Orange** = `ExecutionContext::execute` (GPU kernels; device-side compute)
* **Red** = `cudaMemcpy` (H2D/D2H transfers)
* **Green** = host/device sync (e.g., `cudaStreamSynchronize`); CPU waiting and overlapping with device work — **do not add it as extra GPU time**.
* **E2E latency** = kernels **+** copies **+** necessary sync; this is the primary metric reported in this README.

### Nsight Systems (overview)

* **ONNX**: many long `cudaMemcpyAsync` (red) → dominant E2E cost.
  `images/nsight_onnx_memcpy.PNG`
* **PyTorch**: decode loop stays mostly on GPU (few transfers).
  `images/nsight_pytorch_profile.PNG`
* **TensorRT**: orange (kernels) + red (copies) + green (host waits). **E2E includes kernels and copies**.
  `images/tensorrt_pipeline_latency.PNG`

### ONNX verbose logs (confirm memcpy & fallback)

Representative lines that match what Nsight shows:

```
[W:onnxruntime:, transformer_memcpy.cc:83 ...] 6 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance...
Add Memcpy From Host after <node_name> for CUDAExecutionProvider
Add Memcpy To Host before <node_name> for CUDAExecutionProvider
I: ExecutionFrame ... Using CPUExecutionProvider for <op_name>  (fallback)
```

### Nsight evidence: O1 reduces memcpy & API overhead

> Same settings: `batch=8, seq_len=32, repeat=1`. Same timeline zoom. NVTX ranges aligned.

**Baseline (TensorRT E2E)**
![TensorRT E2E Baseline](images/tensorrt_pipeline_latency.PNG)

*Caption—Baseline:* in-loop `cudaMalloc/cudaFree` are present; red `cudaMemcpy` bars are long and dense because D2H pulls near the full `[B, cur_len, vocab_size]`. Copies dominate the NVTX window.

**O1 (pre-alloc + D2H last step only)**
![TensorRT E2E O1](images/tensorrtO1_pipeline_latency.PNG)

*Caption—O1:* pre-allocation removes per-step `cudaMalloc/cudaFree`; D2H shrinks to just the last-step logits `[B, vocab_size]`. Red `cudaMemcpy` bars are visibly shorter and less frequent; E2E drops by **30–43%** in our tables.

### Where to find the raw artifacts

* Screenshots: [`images/`](images/)
* Nsight Systems report(s): [`nsight_reports/`](nsight_reports/)
* Repro scripts: `run_*` and `benchmark_all.py`


---

## Handling Autoregressive Inference

DistilGPT2 decodes tokens step by step.

* **Attention mask & inputs**: append the new token each step; mask must reflect valid positions.
* **Performance**: sequence grows over time. Without KV cache, each step recomputes the whole sequence.
* **Validation**: compare generated sequences between TensorRT and PyTorch step‑by‑step.

**Sample Autoregressive Loop (Key Logic)**

```python
# Assume device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generated_ids = tokenizer.encode("TensorRT is", return_tensors="pt").to(device)
attention_mask = torch.ones_like(generated_ids).to(device)

for _ in range(max_new_tokens):
    inputs = {"input_ids": generated_ids, "attention_mask": attention_mask}
    logits = tensorrt_model(**inputs).logits  # Or PyTorch/ONNX model
    next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=-1)

output_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

---

## Memory Binding for TensorRT Inference (Pseudo-code Walkthrough)

Efficient inference with TensorRT requires manual memory management for all model inputs and outputs. Unlike higher-level frameworks, TensorRT expects you to allocate GPU memory, manage host-device transfers, set input/output shapes, and bind memory for every inference call.

This section summarizes the **complete memory binding workflow** as annotated pseudo-code, so any engineer can quickly see the core technical steps without digging through long scripts.

---

### Manual Memory Binding: Step-by-Step (Pseudo-code)

```python
# (1) Prepare input data (tokenize & pad to numpy, int32)
input_ids, attention_mask = ...  # [batch, seq], np.int32

# (2) Allocate GPU memory for all inputs/outputs
input_ids_ptr = cudaMalloc(input_ids.nbytes)
attention_mask_ptr = cudaMalloc(attention_mask.nbytes)
output_ptr = cudaMalloc(output_size)

# (3) Copy input data from host (CPU) to device (GPU)
cudaMemcpy(input_ids_ptr, input_ids, cudaMemcpyHostToDevice)
cudaMemcpy(attention_mask_ptr, attention_mask, cudaMemcpyHostToDevice)

# (4) Set dynamic binding shapes for each input (if needed)
context.set_binding_shape(0, input_ids.shape)         # input_ids
context.set_binding_shape(1, attention_mask.shape)    # attention_mask

# (5) Prepare list of all device pointers (bindings)
bindings = [input_ids_ptr, attention_mask_ptr, output_ptr]

# (6) Execute inference (synchronously)
context.execute_v2(bindings)
cudaDeviceSynchronize()

# (7) Copy output data from device back to host
cudaMemcpy(output, output_ptr, cudaMemcpyDeviceToHost)

# (8) Free all device memory (avoid memory leaks)
cudaFree(input_ids_ptr)
cudaFree(attention_mask_ptr)
cudaFree(output_ptr)
```

---

### Key Notes & Best Practices

* **Match all types and shapes**: Input and output arrays must exactly match the engine’s expected shapes and dtypes, or inference will fail.
* **Always free device memory**: Never leave device pointers unfreed—avoid memory leaks, especially in long-running processes or autoregressive loops.
* **Profile shapes in advance**: Dynamic shape support means you must set binding shapes before each inference if batch/seq lengths change.
* **Pre-allocate for efficiency**: For real-time or batch workloads, consider reusing pre-allocated GPU buffers instead of allocating/freeing every step.
* **Reference**: Full working example in `run_tensorrt.py`.


---

## Known Pitfalls & Troubleshooting

The following issues were encountered during this project, with concise explanations and recommended solutions:

### 🔸 \*\* Prefer true end‑to‑end (E2E) over exe\_v2 (kernel-window)\*\*

**Why:**  exe\_v2 (kernel-window) timing omits host↔device transfers (H2D / D2H) and allocation costs. In real deployments these can be as large—or larger—than kernel execution itself, so exe\_v2 numbers often look overly optimistic.

**What to report:** Use **E2E latency** as the primary metric: **H2D → compute → D2H + any required sync** across the entire decode loop.

### 🔸 **Engine Profile Not Covering Autoregressive Sequence Lengths**

* **This is the most common and severe pitfall when deploying autoregressive Transformers with TensorRT.**

* If the TensorRT engine is built with a fixed sequence length (e.g., only `seq_len=12`), **autoregressive decoding**—where the sequence length grows with each generated token—will immediately trigger shape mismatch errors or even `CUDA Error 700 (illegal memory access)` at runtime.

* Example error:

  ```
  Error Code 3: API Usage Error ... Supplied binding dimension [16,5] ... but profile is 12
  Error Code 1: Cuda Runtime (an illegal memory access was encountered)
  ```

* **Solution:**
  Always build the engine with `--minShapes` and `--maxShapes` covering the *entire* range of sequence lengths needed for incremental generation (e.g., `--minShapes=input_ids:1x1 --maxShapes=input_ids:16x32`).
  This ensures all dynamic input shapes used during autoregressive inference are supported by the engine profile.
  *If you forget this, TensorRT will not be able to run incremental generation at all.*

### 🔸 **Unsupported Ops during ONNX Export**

When attempting to export DistilGPT2 using the standard Python `torch.onnx.export` API, I encountered failures due to unsupported operators.

**How I discovered the issue:**

* The export process either crashed or produced ONNX models that failed to run in ONNX Runtime, with error messages indicating missing operator implementations.

**Root cause:**

* Certain operations in DistilGPT2 do not have direct equivalents in standard ONNX, or require special graph optimizations that the default export process cannot handle.

**Resolution:**

* The Hugging Face `transformers.onnx` CLI utility (`python -m transformers.onnx ...`) is specifically optimized for their models.
* Using this tool, ONNX export completed successfully and the generated model could be executed in ONNX Runtime and TensorRT.

**Lesson:**

> When exporting transformer models, always prefer the official `transformers.onnx` export command. If a direct Python export fails, try the CLI utility—these often include custom handling and optimizations for complex transformer operations.

### 🔸 **Autoregressive Input Handling**

* Autoregressive models like DistilGPT2 require careful handling of incremental inference.
* Both `input_ids` and `attention_mask` must be updated step by step (append new token and extend the mask).
* Their dimensions and data types must always match (batch & sequence length, typically int32).
* If they diverge, TensorRT will throw runtime shape errors; if not updated correctly, the model will recompute or mis-handle history.

### 🔸 **trtexec Command Not Found on Windows**

* TensorRT’s CLI tool (`trtexec.exe`) not found in default Windows PATH.
* Solution: Add TensorRT’s `bin/` directory (typically: `C:\TensorRT-8.6.x.x\bin`) to Windows PATH or run directly from that folder.

### 🔸 **TensorRT Engine Profile Constraints & Shape Errors**

* Inputs must fall within the specified min-max range used to build the TensorRT engine.
* Solution:

  * Set appropriate `minShapes`, `optShapes`, and `maxShapes` when building the engine.
  * If encountering "profileMinDims <= dimensions.d\[i]" errors at runtime, rebuild engine with an expanded shape profile or adjust input shape accordingly.

### 🔸 **CUDA Version and Compatibility Issues**

* GTX 1060 (Pascal GPU) supports only up to TensorRT 8.6.x and CUDA 12.x with FP32 precision.
* Solution: Carefully verify GPU compatibility before installation, as newer CUDA/TensorRT versions might not support older GPUs.

### 🔸 **Missing cuDNN or CUDA Dependencies**

* TensorRT execution requires appropriate CUDA/cuDNN libraries; missing libraries cause initialization failures.
* Solution: Install TensorRT bundled with compatible CUDA Toolkit and cuDNN versions. Verify installations via:

```bash
nvcc --version
nvidia-smi
```


---

## What Would Close the Gap 

**Enable a KV cache.** During autoregressive decoding, store past Keys/Values on GPU so each new token only attends to history once (**per-step O(L)**) instead of rebuilding full `t×t` attention (**O(L²)**).

How to implement (pick one):
- **ORT/TRT path with past_key_values**: export `past_key_values` in ONNX and keep a device-resident cache between steps.
- **TensorRT-LLM**: use its built-in KV cache support (plus fused kernels). **If hardware permits**, this is the most straightforward route on newer GPUs.

This one change aligns the algorithmic path with PyTorch `generate` and is the main lever to reach parity on E2E latency.


---

### Key Takeaways
- We report **end-to-end (E2E)** latency, not kernel-only.
- **O1** (pre-alloc + last-step logits only) reduces E2E by **30–43%**.
- PyTorch `generate` is still faster **due to the KV cache (per-step O(L))**; our ORT/TRT path is **O(L²)** without it. **Next step: enable a KV cache** (or move to TensorRT-LLM).



---

## Conclusion

This project is about *how* I:

1. Corrected the metric to **E2E**;
2. Used **Nsight/logs** to find the real bottleneck;
3. Applied a **minimal, effective O1** for **30–43%** E2E reduction;
4. Objectively explained **why TensorRT loses to PyTorch under this path**, and outlined the right next steps.

---

# How to Port a PyTorch→ONNX→TensorRT Pipeline to AMD ROCm Environment

## Overview

This document outlines the key considerations and adaptations required to migrate an existing PyTorch→ONNX→TensorRT deployment pipeline from NVIDIA CUDA to AMD ROCm. While the high-level workflow remains similar, several low-level adjustments are needed due to differences in runtime libraries, kernel optimizations, and supported operators.

---

## 1. Environment Setup

* **NVIDIA:** CUDA Toolkit, cuDNN, TensorRT.
* **AMD:** ROCm stack, MIOpen (analogous to cuDNN), MIGraphX (analogous to TensorRT).

Key difference: NVIDIA provides a tightly integrated toolchain; on AMD, equivalent functionality is spread across ROCm components, requiring careful version alignment.

---

## 2. Model Export (PyTorch → ONNX)

* The PyTorch→ONNX export step remains unchanged.
* Ensure exported operators are supported in MIGraphX; unsupported ops may require custom kernels or graph rewrites.
* Operator coverage is generally improving on ROCm but still lags behind TensorRT.

---

## 3. Inference Runtime (ONNX → Deployment)

* **NVIDIA Path:** ONNX Runtime (CUDA EP) or direct TensorRT engine.
* **AMD Path:** ONNX Runtime (ROCm EP) or MIGraphX runtime.
* Latency characteristics differ: MIGraphX is competitive but may lack some TensorRT-level kernel fusions.

---

## 4. Profiling & Optimization

* **NVIDIA Tools:** Nsight Systems, Nsight Compute, TensorRT Profiler.
* **AMD Tools:** rocProfiler, rocTracer, Radeon GPU Profiler.

Optimization mindset remains the same: analyze hotspots, improve kernel launch efficiency, and ensure memory coalescing. However, profiling workflows differ in tooling and visualization.

## 5. Summary

Porting this pipeline to AMD is straightforward because the overall structure (PyTorch → ONNX → Optimized Runtime → Deployment) stays intact.

The main effort lies in:

* Ensuring the ROCm environment is correctly set up.

* Checking operator compatibility in MIGraphX or ONNX Runtime (ROCm EP).

* Profiling with AMD tools instead of Nsight.

* Retuning kernel execution under HIP when needed.

**Bottom line**: Actual performance tuning on ROCm may require additional investigation, but the migration path conceptually mirrors the NVIDIA flow.

---


## 👨‍💻 Author

**Wang Chen Han**  
5G PHY Algorithm Engineer  
[GitHub: HankWang-WL](https://github.com/HankWang-WL)  
Email: [hank851107@gmail.com](mailto:hank851107@gmail.com)


