<div align="center">

# ⚡ zinfer

**Ultra-fast CPU LLM inference engine written in pure Zig from scratch.**  
*Zero external dependencies · No libc · Direct Linux syscalls · 20–50% faster than llama.cpp*

[![Zig](https://img.shields.io/badge/Zig-0.17.1-orange?logo=zig&logoColor=white)](https://ziglang.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/Dependencies-0%20(no%20libc)-brightgreen)](#architecture)
[![Speedup](https://img.shields.io/badge/vs%20llama.cpp-+20%E2%80%9350%25-success)](#performance--benchmarks)
[![Tests](https://img.shields.io/badge/Tests-27%2F27%20passing-brightgreen)](#testing)

[**English**](README.en.md) · [**Українська**](README.md) · [**Scientific Roadmap**](ROADMAP.md)

</div>

---

## 📌 What is zinfer?

**zinfer** is a modern, minimalist, and deeply optimized Large Language Model (LLM) CPU inference engine for multi-core processors (x86-64 with AVX2 / AVX-512, portable fallback).

Built without any dynamic or static runtime dependencies (`link_libc = false`), it interacts directly with the Linux kernel via raw syscalls and `mmap`, reads **GGUF** model weights, and consistently outperforms `llama.cpp` by **20–50%** in token generation speed through hardware-conscious algorithmic design and statistical physics methods.

```
                    ┌─────────────────────────┐
                    │   GGUF Model (Zero-Copy)│
                    └────────────┬────────────┘
                                 │ mmap
                                 ▼
┌──────────────┐    ┌─────────────────────────┐    ┌─────────────────┐
│ User / API   ├───►│     Runtime State       ├───►│  Token Stream   │
│ CLI / Server │    │  - KV Cache (q2_k / q4) │    │  (Greedy/Tree)  │
└──────────────┘    │  - Thread Pool (Spin)   │    └─────────────────┘
                    └────────────┬────────────┘
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     ▼                           ▼                           ▼
┌─────────────┐           ┌─────────────┐             ┌─────────────┐
│  x8 Repack  │           │ FWHT + PMM  │             │ 2-bit Vocab │
│  GEMM/GEMV  │           │  Attention  │             │   Pruning   │
└─────────────┘           └─────────────┘             └─────────────┘
```

---

## 🚀 Key Features

* 🚫 **Zero Dependencies:** No C/C++ libraries, not even `libc`. Clean binary portability and minimal attack surface.
* ⚡ **x8-Interleaved Weight Repacking:** Proprietary memory packing layout allowing `vpmaddubsw` to produce 8 row dot-products concurrently with zero horizontal reductions.
* 🧠 **Comprehensive Architecture Support:**
  * **Google Gemma 4:** Full support for PLE (gated residuals), dual RoPE (SWA / Global), normalization layers, and shared KV projections.
  * **Google Gemma 3:** 270M–27B, FunctionGemma, GeGLU, sandwich norms, QK-norm.
  * **Liquid LFM-2, LFM-2.5, LFM-3:** Short-conv + GQA hybrid layers, sandwich norms, SWA.
  * **Llama 2 / 3 / 3.1 / 3.2 / Mistral / TinyLlama:** Full LLaMA stack with `rope_freqs`.
  * **Qwen 2 / Qwen 3:** Full support including reasoning think blocks (`--no-think`).
  * **MoE (Mixture of Experts):** Qwen2-MoE, Mixtral 8x7B, DeepSeek-V2 with thread-pool expert routing.
* 📦 **Extreme KV Cache Compression:**
  * **q2_k (Hybrid 2-bit QuaRot):** 2-bit $K$ + 4-bit $V$ with FWHT rotation. **4.57× memory reduction** with excellent attention fidelity (cosine similarity **0.9822**).
  * **q2_0 (Pure 2-bit):** 6.4× memory compression.
  * **q4_0 / q8_0 / f16:** Standard cache formats supported.
* 🔬 **Mathematical Innovations (PMM & FWHT):**
  * **Fast Walsh-Hadamard Transform (FWHT):** $O(d \log d)$ SIMD rotation of $Q$ and $K$ eliminates outlier channels, improving quantization SQNR by **+3.74 dB** (57.7% MSE reduction).
  * **Kunchenko Polynomial Maximization Method (PMM):** High-order non-Gaussian cumulant estimators ($c_2, c_3, c_4$) and asymmetric shift estimator $\hat{\theta}$ minimize quantization clipping error.
  * **Predictive FFN Sparsity:** Zero-overhead skipping of dead neurons in GeLU/SwiGLU saves up to 12% memory bandwidth during prefill.
  * **Attention Noise Floor Cutoff:** Softmax tail pruning drops irrelevant context tiles early (+31% prompt processing).
* 🌳 **Tree-based Speculative Decoding:**
  * **Tree of Hypotheses:** Prefix tree of multiple continuation candidates; instantaneously recovers on alternative branches when primary draft diverges (up to **3.4× speedup**).
  * **Draft Model Support:** Small draft model speculation over identical vocabularies.
  * **Adaptive Draft Controller:** Dynamic adjustment of draft length $K \in [1..8]$ using Shannon logit entropy.
* 🌐 **StreamingLLM & NTK RoPE:**
  * 4 Attention Sinks for infinite conversation context streaming.
  * Dynamic NTK-aware RoPE frequency scaling for handling extended context windows beyond training limits.
* 🖥️ **OpenAI-Compatible HTTP Server:**
  * Endpoints `/v1/chat/completions` and `/v1/responses` for out-of-the-box integration with Open WebUI, Ollama, and LangChain.
  * Persistent prefix KV-cache and Server-Sent Events (SSE) streaming.

---

## 🛠️ Quickstart

### Building from Source

Requires [Zig](https://ziglang.org/download/) **0.17.1**.

```sh
# Clone repository
git clone https://github.com/pylypenko99/zinfer.git
cd zinfer

# Build optimized binary (auto-detects AVX2 / AVX-512 / FMA)
zig build -Doptimize=ReleaseFast

# Run test suite (27 unit tests)
zig test -OReleaseFast -mcpu=native src/tests.zig
```

The resulting binary will be placed at `zig-out/bin/zinfer`.

---

## 💻 Usage Examples

### 1. Basic Generation
```sh
./zig-out/bin/zinfer -m models/Llama-3.2-1B-Instruct.gguf -p "Explain quantum computing in simple terms:" -n 128 --temp 0.7
```

### 2. Interactive Chat
```sh
./zig-out/bin/zinfer -m models/gemma-4-E2B-it.Q4_K_M.gguf --chat -s "You are an expert Zig systems programmer."
```

### 3. Tree-based Speculative Decoding
Accelerates code and structured syntax generation up to 3.4× by mining context patterns:
```sh
./zig-out/bin/zinfer -m models/LFM-3-Coder-1.7B.Q4_K_M.gguf -p "fn quicksort(comptime T: type" --draft 5
```

### 4. KV Cache Quantization (Up to 4.57× RAM savings)
```sh
# Hybrid 2-bit K / 4-bit V cache (optimal fidelity and memory reduction)
./zig-out/bin/zinfer -m models/gemma-4-E2B-it.Q4_K_M.gguf --cache q2_k -p "Hello world"

# 4-bit symmetric cache
./zig-out/bin/zinfer -m models/Llama-3.2-1B.gguf --cache q4_0 -p "Hello world"
```

### 5. Local OpenAI-Compatible Server
```sh
./zig-out/bin/zinfer -m models/Qwen3-0.6B-Q8_0.gguf --server --port 8080
```
Test via `curl`:
```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Hello! Write a hello world in Zig."}],
    "temperature": 0.2
  }'
```

---

## 📊 Performance & Benchmarks

### Intel Core i7-8550U (4C/8T, DDR4 ~28 GB/s peak, AVX2)
Measured directly against `llama.cpp` in CPU-only mode (`-ngl 0`):

| Model | zinfer pp256 (tok/s) | llama.cpp pp256 | zinfer tg64 (tok/s) | llama.cpp tg64 | zinfer Advantage |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Llama-3.2-1B Q4_K_M** | **194.0** | 154.0 | **38.5** | 27.0 | **+42.6%** |
| **Qwen3-0.6B Q8_0** | **243.0** | 130.0 | **48.1** | 31.5 | **+52.7%** |
| **Qwen3-0.6B Q4_0** | **308.0** | 184.0 | **68.4** | 50.6 | **+35.2%** |
| **LFM-3-Coder-1.7B Q4_K_M** | **128.0** | 107.0 | **27.8** | 18.4 | **+51.1%** |
| **Gemma-3-1B-It Q5_0** | **208.0** | 105.0 | **43.4** | 26.1 | **+66.3%** |

### AMD Ryzen 7 9700X (8C/16T, Zen 5, DDR5, AVX2/AVX-512)
Measured directly against `llama.cpp` (AVX-512 VNNI):

| Model | zinfer pp256 | llama.cpp pp256 | zinfer tg64 | llama.cpp tg64 | zinfer Advantage |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Llama-3.2-1B Q4_K_M** | **941.0** | 708.0 | **77.9** | 59.7 | **+30.5%** |
| **LFM3-1.7B Q4_K_M** | **610.0** | 465.0 | **53.8** | 44.1 | **+22.0%** |

---

## 🗂️ KV Cache Compression Comparison

| Format | Bytes / elem | Compression | RAM for Gemma 4 (4K ctx) | Attention Similarity | Application |
|---|:---:|:---:|:---:|:---:|---|
| **FP16** | 2.00 | 1.0× | 30.2 MB | 1.0000 | Reference baseline |
| **Q8_0** | 1.06 | 1.9× | 16.1 MB | 0.9998 | Indistinguishable quality |
| **Q4_0** | 0.56 | 3.6× | 8.5 MB | 0.9929 | Optimal performance |
| **Q2_K (Hybrid)** | **0.44** | **4.57×** | **6.6 MB** | **0.9822** | **QuaRot + FWHT, zero V noise** |
| **Q2_0 (Pure)** | 0.31 | 6.4× | 4.7 MB | 0.8413 | Extreme compression |

---

## 🏗️ Architecture & Codebase

The codebase is split into modular components without cyclic dependencies:

```
src/
├── main.zig        # CLI entry point, chat loop, benchmark runner
├── server.zig      # Zero-libc HTTP server (OpenAI API, SSE streaming)
├── model.zig       # Core transformer model state and forwardBatch
├── attention.zig   # FlashAttention, KV-cache quantization, FWHT, PMM kernels
├── ffn.zig         # Activation functions (SiLU, GeLU), SwiGLU, MoE top-K routing
├── rope.zig        # RoPE frequency precomputation, NTK scaling, 2D rotary embedding
├── repack.zig      # x8 matrix weight repacking, SIMD GEMM/GEMV microkernels
├── quant.zig       # Block quantization types (Q4_0..Q6_K, Q8_0, Q8_K)
├── pool.zig        # Lock-free thread pool with two-phase spin/futex waiting
├── gen.zig         # Autoregressive generation loop, Tree Speculation controller
├── sampler.zig     # SIMD argmax, temperature, top-k/top-p, min-p sampler
├── tokenizer.zig   # Byte-BPE / SentencePiece with min-heap priority queue & Unicode
└── tests.zig       # Unified test runner (27 tests)
```

---

## 🧪 Testing

Deterministic token-for-token verification is performed across architectures:
* Greedy decoding (`--temp 0`) strictly matches `llama.cpp` on Llama-3.2, Gemma 4, Qwen3, and LFM3.
* All 27 unit tests pass with zero warnings:

```sh
$ zig test -OReleaseFast -mcpu=native src/tests.zig
All 27 tests passed.
```

---

## 📜 License

Distributed under the [MIT License](LICENSE). Open-source and free for both research and commercial applications.
