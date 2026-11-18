# zinfer — план виконання для AI-агентів

> Джерело: аналіз README/ROADMAP/src станом на 18.11.2026 (репо pylypenko99/zinfer, гілка main).
> Формат — **тікети для виконання AI-агентами**. Кожен тікет самодостатній: контекст (файли), кроки, тест-модель, DoD. Порядок у межах фази = порядок залежностей. P1-A і P2-A можна виконувати паралельно.
> Супутні документи в `plan/`: `low-ram-moe.md` (P4-C), `zquant-quantizer.md` (референс ядер для P2-A), `code-review-2026-11.md` (застаріле ревʼю — джерело ідей, звіряти з кодом).

## 0. Спільні правила (порушення = тікет не приймається)

**Збірка/тести/перевірка** (Zig 0.17.1):
```sh
zig build -Doptimize=ReleaseFast                    # збірка
zig test -OReleaseFast -mcpu=native src/tests.zig   # усі тести мають лишатися зеленими
./zig-out/bin/zinfer -m <model.gguf> -p "..." -n 64 --temp 0   # greedy-паритет vs llama.cpp
```

**Незмінні інваріанти:**
1. Zero-deps, `link_libc = false` — жодних нових залежностей/системних бібліотек.
2. Нуль алокацій у decode-циклі (тільки при init/repack).
3. Жодна з підтримуваних архітектур не деградує: greedy-паритет з llama.cpp зберігається.
4. Perf-гейт: tg64 на реперних моделях (Llama-3.2-1B Q4_K_M, Qwen3-0.6B Q8_0) не гірше ніж −3%.
5. Оновити README (таблиця supported + perf-рядок) та tests.zig (новий golden на кожен тікет).

**Карта модулів:** model.zig (Model/State/біндери тензорів) · attention.zig (KV, FWHT, PMM, mlaSplitKv/mlaScore) · ffn.zig (активації, MoE routeTopK) · rope.zig (частоти, NTK) · repack.zig (x8-інтерлівінг, vpmaddubsw) · quant.zig (типи ваг/активацій) · pool.zig (spin+futex) · gen.zig (драфт/дерево) · sampler.zig · tokenizer.zig · grammar.zig · server.zig · gguf.zig · platform.zig (huge pages, syscalls).
**Джерело правди щодо GGUF-імен тензорів/метаданих** — convert-скрипти llama.cpp + ggml.h.

## 1. Поточний стан (факти з репо)

**База:** Zig 0.17.1, zero-deps, `link_libc=false`, прямі syscalls, mmap GGUF zero-copy, бінарник 12 МБ, CI + release-воркфлоу (v3/v4 бінарники).

**Архітектури (`Arch` enum у model.zig):** `llama, qwen2, qwen3, lfm2, lfm3, gemma3, gemma4, qwen2moe, mixtral, deepseek2`.
- Gemma4: PLE (per-layer embeddings), дуальний RoPE (SWA/Global), shared-KV шари, key_length_swa, n_rot 128/256.
- LFM2/3: short-conv гібрид, SWA, per-head sigmoid gate (`attn_gate`), sandwich norms.
- MoE: `qwen2moe`, `mixtral` — `ffn_*_exps` 3D-тензор + окремі експерти, shared expert, router.
- QK-norm, q/k/v bias, attn_scale для Gemma3-27B — усе є.

**Формати ваг (quant.isSupported):** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K.
**Repack x8 (vpmaddubsw):** q4_0, q4_1, q5_0, q5_1, q8_0, q4_k, q5_k, q6_k; huge-pages регіон для ваг.

**KV-кеш:** тайли по 16 позицій; f16 / q8_0 / q4_0 / q2_0 / **q2_k-гібрид (2-bit K + 4-bit V + FWHT QuaRot, cos 0.9822)**; SWA-ринг (window-тайли без маски); flash-decoding (чанки+reduce); shared-KV слоти.

**Матеметоди:** FWHT SIMD; ПММ Кунченка (кумулянти c2..c4, асиметрія γ3, зсув θ̂) у quantBlock32; predictive FFN sparsity (dead-блоки ≤−4/−6 → пропуск W_up, +12% pp); attention noise floor (тайли max(s)<m−13 → пропуск V, +31% pp).

**Декодування:** greedy/семплер (top-k/top-p/min-p), entropy-адаптивний драфт K∈[1..8], Tree of Hypotheses (альтернативні гілки без reset), draft-модель, rewind KV після reject; vocab pruning 2-бітний coarse lm_head (двофазний).

**Інше:** NTK-aware RoPE; StreamingLLM sinks; grammar/JSON logit-mask (`--json`); OpenAI-сумісний сервер (`/v1/chat/completions`, `/v1/responses`, SSE, персистентний префіксний KV); BPE+SentencePiece токенайзер (min-heap, O(log K) unicode-таблиці).

**Бенчмарки (з README):** i7-8550U: +33..46% tg64 vs ik_llama.cpp; Ryzen 9700X: +22..30% vs llama.cpp; ефективність вичитки 88–92% піку RAM; TTFT 39.8 мс.

## 2. Прогалини до «усіх типів LLM»

### 2.1 Архітектури
| Статус | Що |
|---|---|
| ❌ Блоковано | **MLA e2e (DeepSeek-V2/V3/R1)**: ядра `mlaSplitKv`/`mlaScore` є, але `attn_kv_a_mqa` у біндері → `UnsupportedArchitecture`. Головний пріоритет. |
| ❌ Відсутня | **gpt-oss**: MoE + MXFP4-ваги, гібридний SWA 128 local/128 global (чергування), sinks (уже є!), SwiLU, per-layer norms. |
| ❌ Відсутня | **Phi-3/3.5/4**: long-rope (factor-таблиці), частковий rotary (n_rot < head_dim — частково є), GeLU (є). |
| ❌ Відсутні | Qwen3-Next (Gated DeltaNet), Qwen3-VL/мультимодальне (mrope + vision), RWKV-6/7, Mamba/SSM, GPT-2/NeoX/J legacy, OLMo, StableLM, Command-R, Gemma2 (softcap у attention — фінальний уже є), Granite, MiniCPM, Exaone. |
| ⚠️ Перевірити | Qwen3-MoE (30B-A3B): шлях `qwen2moe` близький — потрібен тест + можливо аліас `qwen3_moe` з QK-norm у MoE-шарах; tied embeddings (3.2-1B) — наче є (`output.weight` optional). |
| ⚠️ RoPE scaling | реалізовано лише `linear`; потрібні **yarn** (llama3.1/Qwen long-ctx) і **llama3-freq** — зараз warn + деградація. Phi потрібні longrope-фактори (тензори `rope_factors_long/short`). |

### 2.2 Формати ваг (для макс. стиснення моделей)
Відсутні: **Q2_K, Q3_K** (ваги), усі **IQ\*** (IQ4_XS/NL, IQ3, IQ2, IQ1_S/M), **TQ1_0/TQ2_0** (BitNet), **MXFP4/NVFP4** (gpt-oss), F8_E4M3. → природня точка інтеграції з zquant (див. `plan/zquant-quantizer.md`).

### 2.3 Продуктивність (горизонти з ROADMAP + нові)
- AVX-512 ZMM 16-токенні ядра (tok_group 8→16, регістровий бюджет Zen5) — у ROADMAP.
- Batch-верифікація дерева гіпотез за один forward — у ROADMAP.
- Continuous batching у сервері (зараз персистентний префіксний KV; потрібні paged-KV + динамічний набір послідовностей).
- NUMA-пінг (set_mempolicy без libc через syscall), multi-socket розшарування MoE-експертів.
- aarch64: NEON/dotprod-ядра (x8-repack у vpmaddubsw — x86-only; потрібен порт або авто-вибір plain-ядер).
- KV q2_k для всіх арх (зараз FWHT-вимога hd — степінь 2).

## 3. Тікети (порядок = порядок виконання)

Загальний DoD будь-якого тікету: збірка ✓, `zig test` ✓, greedy-паритет з llama.cpp на вказаній моделі ✓, README/tests оновлені ✓, perf-гейт −3% ✓.

### Фаза P1 — найбільший користувацький виграш

**P1-A · MLA e2e для DeepSeek (розмір L, залежностей немає)**
- Контекст: `model.zig` — у `bindTensors` рядок `attn_kv_a_mqa → UnsupportedArchitecture`; `attention.zig` — готові ядра `mlaSplitKv`/`mlaScore`.
- Кроки: (1) взяти імена MLA-тензорів з convert-скриптів llama.cpp (q_a/q_b/q_a_norm, kv_a_mqa/kv_b/kv_a_norm, k_rope-компонента); (2) розширити `Attention` опціональними MLA-тензорами, прибрати блокування в біндері; (3) latent-KV (d_c + rope d_r) у існуючі KV-слоти (новий CacheType або формат слота); (4) forward: q_a→RMSNorm(q_a_norm)→q_b→head-split; kv_a_mqa→norm→kv_b (wkv u/kv), rope на k_rope; (5) зʼєднати з `mlaSplitKv`/`mlaScore` у attnWorker.
- Тест-модель: **DeepSeek-V2-Lite** (GGUF Q8_0/Q4_K); далі дистиляти DeepSeek-R1-1.5B/7B.
- DoD: greedy token-у-токен == llama.cpp на 3 промптах; ppl(wiki.test.raw) Δ<0.01; рядок у README.

**P1-B · RoPE-scaling yarn + llama3 (розмір M)**
- Контекст: `rope.zig` (`ropeFreqs`, `ntkAwareBase`), `model.zig` (гілка `rope.scaling.type` з warn).
- Кроки: yarn (factor/ms/ms_all_dim з метаданих, відфільтр частот), llama3-freq (секції low/high factor); вибір типу за `rope.scaling.type`; юніт-тести частотних таблиць проти чисел llama.cpp.
- Тест-модель: Llama-3.1-8B (контекст 16k+), Qwen2.5-7B-Instruct.
- DoD: жодного warn «not implemented»; ppl на 16k не деградує; контекст >ctx_train працює через NTK+yarn.

**P1-C · Qwen3-MoE (розмір S/M)**
- Контекст: `model.zig` (аліас `qwen2_moe`→Arch, MoE-біндер `ffn_*_exps`), QK-norm (`optVector`) вже є.
- Кроки: приймати arch `qwen3_moe` → той самий шлях; перевірити QK-norm у MoE-шарах, shared expert, top-k; за потреби — no-bias шляхи.
- Тест-модель: Qwen3-30B-A3B (Q4_K_M).
- DoD: greedy parity; tg64 ≥ llama.cpp×0.8 на MoE (роутинг паралельний через пул).

### Фаза P2 — нижчий bpw + gpt-oss (P2-A незалежний від P1, можна стартувати паралельно)

**P2-A · Формати ваг Q2_K/Q3_K/IQ\* (розмір L)** — джерело ядер: `plan/zquant-quantizer.md` (блоки, LUT-гріди, comptime-диспетчер).
- Кроки: `quant.zig` — нові GgmlType + `dequantRow`/`dotN`; `repack.zig` — x8-варіанти (спершу без repack: plain-ядра); обовʼязково comptime-перевірки `@sizeOf` блоків.
- Порядок: Q2_K, Q3_K → IQ4_XS/IQ4_NL → IQ3_XXS → IQ2_XS/S/M → IQ1_S/M.
- Тест-моделі: bartowski/unsloth GGUF відповідних типів (Qwen3-0.6B, SmolLM2-360M).
- DoD: пряме завантаження IQ-GGUF з HF; tg64 ≥ llama.cpp на тій самій моделі/кванті; tests.zig — round-trip dequant.

**P2-B · gpt-oss (розмір L, залежить від P2-A для MXFP4-ядер)**
- Кроки: арх `gptoss`; MXFP4-ядра (E8M0 масштаб + E2M1 значення, блок 32 — layout у zquant-док.); SWA-чергування 128/128 через `sliding_window_pattern` bool[] (інфраструктура gemma4); attention sinks (StreamingLLM — вже є); SwiLU; MoE top-4 + normalization масштабів.
- Тест-модель: gpt-oss-20b.
- DoD: greedy parity; ppl Δ<0.02 проти llama.cpp (MXFP4).

**P2-C · TQ2_0/TQ1_0 BitNet (розмір S, після P2-A)** — ядра з zquant-док.; тест BitNet-b1.58-2B GGUF; DoD: завантаження+генерація.

### Фаза P3 — довгий хвіст (P3-A/B незалежні; P3-C після P3-A через mrope)

**P3-A · Phi-3/3.5/4 (розмір M)**: long-rope фактори (тензори `rope_factors_short/long`), частковий n_rot; тест Phi-4-mini / Phi-3.5-mini. DoD: greedy parity.
**P3-B · Legacy/long-tail (розмір S кожен, окремі PR)**: gemma2 (attn-logit softcap), OLMo/OLMo2, StableLM, Command-R, GPT-2/NeoX/J (LayerNorm + parallel residual), MiniCPM, Granite — зівставити список архів llama.cpp і закрити порожні; кожен = enum + біндер + golden.
**P3-C · Мультимодальне (розмір L)**: mrope (секційні позиції у `rope.zig`), vision-projector/mmproj GGUF, `--image` у CLI; тест Qwen3-VL / Gemma3-VL.

### Фаза P4 — SSM + максимальний perf

**P4-A · SSM-мікшери (розмір L)**: новий варіант union mixer: Mamba-2 selective scan → RWKV-6/7 (wkv6/wkv7-референс у ggml) → Qwen3-Next GatedDeltaNet; каркас стану — як shortconv LFM.

**P4-B · Perf-пакет (кожен розмір M/L, незалежні):**
1. ZMM 16-токенні ядра (tok_group=16 на AVX-512VL Zen5; не регресити AVX2 — умовна компіляція як зараз).
2. Batch-верифікація дерева гіпотез одним forward (umbrella-batch у gen.zig).
3. Continuous batching у server.zig: paged-KV (блоки по 16 — тайли вже є), динамічний набір послідовностей, планувальник prefill/decode.
4. NUMA: syscall set_mempolicy без libc (`platform.zig`), пінг потоків пулу.
5. aarch64: NEON-dotprod ядра repack/plain + авто-вибір цільових бінарників у release-воркфлоу.
- DoD perf: tg64 ≥90% RAM-BW на 2 реперних моделях; сервер ≥4 конкурентні послідовності без деградації >20%.

**P4-C · Low-RAM MoE (розмір L)** — повна готова спека: `plan/low-ram-moe.md` (x8cache на диску + clean file-backed pages + MADV_DONTNEED eviction + три ешелони prefetch-стрімінгу експертів). Тест: Qwen3-30B-A3B у RAM ≤16 ГБ. DoD: інференс без OOM; tg64 на NVMe ≥40% від pure-RAM режиму; жодних записів у swap.

## 4. Продуктивність: правила і бюджет

- Головна метрика decode: GB/с вичитки ваг (тримати 88–92% → ціль 94% на ZMM-ядрах) і tokens/s на фіксованому зоопарку.
- Бенчмарк-таблиця README оновлюється автоматично CI (жодних ручних чисел).
- Perf-регресійний гейт: −3% tokens/s на будь-якій реперній моделі блокує merge.
- Кожна нова архітектура — одразу рядок у perf-таблицю (pp512/tg64, 2 машини: AVX2-лаптоп + Zen5).

## 5. Тестування

- Розширити tests.zig (27 → golden-набір на кожен арх-тип: конфіг-парсинг, біндери, rope-таблиці).
- Швидка перевірка архітектури — розд. 5.1: на кожен Arch-тип призначена модель (де існує ≤1B); ≤1B = обовʼязковий smoke на кожен PR (хвилини), решта — нічні.

### 5.1 Швидка перевірка: модель на кожен арх-тип (≤1B де існує)

| Арх (Arch enum / тікет) | Модель для швидкої перевірки | Парам. | Джерело GGUF |
|---|---|---|---|
| llama | **SmolLM2-360M-Instruct** (осн.) / **SmolLM2-135M** (CI-мінімум) | 0.36B / 0.13B | bartowski/SmolLM2-*-GGUF |
| qwen2 | **Qwen2.5-0.5B-Instruct** | 0.5B | bartowski |
| qwen3 | **Qwen3-0.6B** | 0.6B | bartowski |
| lfm2 | **LFM2-350M** (або LFM2.5-350M — та сама арха) | 0.35B | LiquidAI/LFM2-350M-GGUF |
| lfm3 | ⚠️ **LFM3-1.7B** (найменша серії; ~1 ГБ у Q4_K_M) | 1.7B | LiquidAI |
| gemma3 | **Gemma-3-270M-it** | 0.27B | bartowski / unsloth |
| gemma4 | ⚠️ **Gemma-4-E2B** (2.1B params — найменша серії) | 2.1B | bartowski |
| qwen2moe | ⚠️ **Qwen1.5-MoE-A2.7B-Chat** (найменша qwen2_moe; 14B total, 2.7B active) | 14B | bartowski |
| mixtral | ⚠️ **Mixtral-8x7B** (найменша серії) | 47B | ggml-org |
| deepseek2 (P1-A) | ⚠️ **DeepSeek-V2-Lite-Chat** (найменша MLA; 16B, 2.4B active) | 16B | bartowski |
| qwen3_moe (P1-C) | ⚠️ **Qwen3-30B-A3B** (найменша qwen3-moe) | 30B | bartowski / unsloth |
| gptoss (P2-B) | ⚠️ **gpt-oss-20b** (найменша серії) | 21B | ggml-org/gpt-oss-20b-GGUF |
| phi (P3-A) | ⚠️ **Phi-3.5-mini-instruct** (найменша Phi-3.x); legacy-гілка менша: **Phi-1.5** | 3.8B / 1.3B | bartowski |
| gemma2 (P3-B) | ⚠️ **Gemma-2-2b-it** (найменша серії) | 2.6B | bartowski |
| gpt2 (P3-B) | **GPT-2-124M** ✓ (legacy-шлях: LayerNorm, learned pos) | 0.12B | конверт convert_hf_to_gguf.py |
| olmo (P3-B) | **OLMo-2-0425-1B** ✓ (якщо GGUF нема — конверт) | 1B | allenai → конверт |
| granite (P3-B) | ⚠️ **granite-3.3-2b-instruct** (найменша dense; гібрид 3.0-1b-a400m — у рядку mamba) | 2B | ibm-granite (IBM/gguf) |
| stablelm (P3-B) | ⚠️ **stablelm-2-1_6b-chat** (найменша серії) | 1.6B | bartowski |
| command-r (P3-B) | ⚠️ **Command-R7B** (c4ai-command-r7b-12-2024 — найменша R-серії) | 7B | dranger003-GGUF |
| mamba/SSM (P4-A) | **Mamba-370M** ✓ + **Granite-3.0-1B-A400M** ✓ (гібрид Mamba2+MoE) | 0.37B / 1B | конверт; ibm-granite |
| rwkv (P4-A) | ⚠️ **RWKV-6-World-1B6** (найменша GGUF-готова; є новіші RWKV7-G1) | 1.6B | BlinkDL / bartowski |
| TQ/BitNet (P2-C) | ⚠️ **BitNet-b1.58-2B4m** (найменша тернарна) | 2.4B | ggml-org/BitNet-b1.58-2B4M-GGUF |
| мультимодальне (P3-C) | ⚠️ **Qwen2-VL-2B-Instruct** (найменша VL; mrope-референс) | 1.5B | bartowski |

**Правила швидкої перевірки:**
- ✓ (≤1B): smoke на кожен PR — завантаження, greedy `-n 64 --temp 0` (паритет з llama.cpp), ppl на1 чанку wiki.test.raw; бюджет ≤5 хв CPU, RAM ≤4 ГБ.
- ⚠️ (серія не має ≤1B): вказана **найменша існуюча версія серії** (звірено по HF станом на 18.11.2026); нічний прогін, Q4_K_M/Q8_0 замість F16 (швидше у2–4 рази), `-n 64`.
- Кожен GGUF має читатись обома: llama.cpp і zinfer (це перший тест тікета).
- Кеш моделей CI: `~/.cache/zinfer-models` (прямі лінки HF, без LFS).
- Параметри тест-моделей для тікетів P1–P4 у розд.3 посилаються на цю таблицю.
- Паритет: greedy token-у-токен vs llama.cpp (вже практикується) + ppl на wiki.test.raw (дельта <0.01 на Q4_K_M).
- Токенайзер: golden пари HF для BPE/SP на кожну архітектуру.

## 6. Ризики

| Ризик | Мітигація |
|---|---|
| MLA: два шари проекцій + rope на латенті | точна версія → паритет → fuse; маленька модель V2-Lite для CI |
| gpt-oss hybrid SWA | використати наявний sliding_window_pattern[] + sinks; юніт чергування |
| IQ-ядра повільніші K-квантів на AVX2 | repack-адаптація; benchmark per-тип; документувати, де падіння |
| Qwen3-Next/DeltaNet — новий тип мікшера | спершу reference-реалізація, потім fused; патерн shortconv як каркас стану |
| Low-RAM: холодні сторінки → stall на хвіст latency NVMe | ешелони перекриття з low-ram-moe.md; вимірювати p99 ttft |
| Регресії швидкості від нових гілок | perf-гейт у CI (−3%) |
| Розрив документації (27 vs 32 тести) | тест-каунтер генерувати з коду |

## 7. Референси

- convert-скрипти llama.cpp (`convert_hf_to_gguf.py`) — імена тензорів/метаданих по архітектурах.
- `ggml.h` / `ggml-common.h` — layout-и блоків, enum ggml_type (TQ1_0=34, TQ2_0=35, MXFP4=39, NVFP4=40, Q1_0=41, Q2_0=42, F8_E4M3=43).
- `plan/zquant-quantizer.md` — референс ядер квантування для P2-A/P2-B/P2-C.
- `plan/low-ram-moe.md` — спека P4-C.
- `plan/code-review-2026-11.md` — історичне ревʼю (звіряти з поточним кодом).
