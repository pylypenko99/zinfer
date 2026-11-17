# 📢 Матеріали для анонсу zinfer в X (Twitter)

Тут зібрані готові варіанти публікацій для соціальної мережі X (Twitter) для презентації проєкту та залучення аудиторії, розробників і зірок на GitHub.

---

## Варіант 1: Англомовний детальний тред (Рекомендований для світової спільноти)

### 🧵 Tweet 1 (Hook / Заголовок)
What if LLM inference on CPU didn't need C++, external runtimes, or even libc?

Meet **zinfer** ⚡: an ultra-fast LLM inference engine written from scratch in pure @ziglang (0.17.1).

🚀 20–50% faster than llama.cpp on identical hardware
📦 0 dependencies · Direct Linux syscalls
🧠 Gemma 4, MoE, LLaMA 3.2, LFM3

🔗 https://github.com/pylypenko99/zinfer

🧵👇

---

### 🧵 Tweet 2 (Архітектура та x8 Repacking)
How is it 20–50% faster than llama.cpp?

1/ **x8 Weight Repacking**: Matrices are interleaved during load so one 32-byte vector packs 4 weights across 8 rows.
`vpmaddubsw` produces 8 row dot-products simultaneously — zero horizontal reductions!

2/ **2-Stage Vocab Pruning**: 2-bit Lloyd-Max candidate filter accelerates `lm_head` by 10–15× on 128k+ vocabularies.

---

### 🧵 Tweet 3 (Гібридний 2-бітний KV-Кеш + FWHT + ПММ)
KV Cache scaling without quality loss:

🔬 **Fast Walsh-Hadamard Transform (FWHT)** rotates Q and K in $O(d \log d)$, eliminating activation outliers (+3.74 dB SQNR).
🎯 **PMM (Kunchenko Polynomials)**: higher-order cumulant & skewness shift estimators adaptively minimize clipping error.
📦 **Hybrid q2_k**: 2-bit K + 4-bit V achieves **4.57× RAM compression** with 0.9822 cosine attention fidelity!

---

### 🧵 Tweet 4 (Підтримка моделей)
Built for modern LLM architectures:

• **Google Gemma 4**: PLE gated residuals, dual SWA/Global RoPE, shared KV layers
• **MoE**: Mixtral 8x7B, Qwen2-MoE, DeepSeek-V2 with lock-free thread pool routing
• **Liquid LFM-2/3**: 13 short-conv + 11 GQA hybrid layers
• **Llama 2/3.x, Qwen 2/3, Mistral**

All greedy outputs verified token-for-token against llama.cpp.

---

### 🧵 Tweet 5 (Tree of Hypotheses Speculation)
Need more speed?

🌳 **Tree of Hypotheses**: Generates an n-gram prefix tree of candidate continuations. If the primary prediction diverges, it immediately switches to an alternative valid branch without losing draft progress.

Result: up to **3.4× speedup** on code and structured JSON syntax!

---

### 🧵 Tweet 6 (OpenAI API Server & Call to Action)
zinfer includes a zero-libc, embedded HTTP server compatible with OpenAI `/v1/chat/completions`, SSE streaming, and persistent prefix caching.

Open source under MIT:
⭐ Star the repo & test it on your CPU:
👉 https://github.com/pylypenko99/zinfer

#ziglang #ai #llm #opensource #machinelearning #systems

---

## Варіант 2: Лаконічний окремий пост (Single Viral Post — English)

Meet **zinfer** ⚡: an ultra-fast LLM CPU inference engine written in 100% pure @ziglang (0.17.1) with ZERO dependencies — no C/C++, not even libc!

🔥 20–50% faster than llama.cpp on x86-64
🧠 Gemma 4, MoE (Mixtral/Qwen), LLaMA 3.2, LFM-3
📦 4.57× KV Cache compression via Hybrid 2-bit QuaRot + FWHT
🌳 Tree-based Speculative Decoding (up to 3.4× speedup)
⚡ OpenAI-compatible edge server with prefix caching

Check out the code & benchmarks:
⭐ https://github.com/pylypenko99/zinfer

#ziglang #llm #ai #opensource

---

## Варіант 3: Україномовний анонс (Для української tech-спільноти)

Презентуємо **zinfer** ⚡ — надшвидкий рушій інференсу LLM на CPU, написаний з нуля на чистому @ziglang без жодної сторонньої залежності (навіть без libc)!

Чому zinfer перевершує аналоги:
🚀 **На 20–50% швидший за llama.cpp** на ідентичному залізі (i7 DDR4 та Ryzen 9700X DDR5).
🧠 **Повна підтримка моделей:** Google Gemma 4 (PLE, shared KV), MoE (Mixtral, Qwen2-MoE), Liquid LFM-3, LLaMA 3.2.
📦 **Гібридний 2-бітний KV-кеш (q2_k):** 4.57× компресія пам'яті на базі FWHT та поліноміальних оцінювачів ПММ (Кунченка) при збереженні 0.9822 косинусної схожості уваги.
🌳 **Деревоподібне спекулятивне декодування (Tree of Hypotheses):** до 3.4× прискорення генерації коду.
🖥️ Вбудований zero-libc HTTP-сервер, сумісний з OpenAI API (`/v1/chat/completions`) та стрімінгом.

Проєкт повністю open-source (MIT). Підтримайте зіркою на GitHub! ⭐
👉 https://github.com/pylypenko99/zinfer

#ziglang #ai #llm #opensource #ua_tech
