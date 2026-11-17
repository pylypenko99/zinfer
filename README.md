<div align="center">

# ⚡ zinfer

**Надшвидкий інференс LLM на CPU, написаний на чистому Zig з нуля.**  
*Zero dependencies · No libc · Прямі системні виклики · 20–50% швидше за llama.cpp*

[![Zig](https://img.shields.io/badge/Zig-0.17.1-orange?logo=zig&logoColor=white)](https://ziglang.org/)
[![CI](https://github.com/pylypenko99/zinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/pylypenko99/zinfer/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/pylypenko99/zinfer?logo=github&color=blue)](https://github.com/pylypenko99/zinfer/releases/latest)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/Dependencies-0%20(no%20libc)-brightgreen)](#архітектура)
[![Speedup](https://img.shields.io/badge/vs%20llama.cpp-+20%E2%80%9350%25-success)](#швидкість-і-бенчмарки)
[![Tests](https://img.shields.io/badge/Tests-32%2F32%20passing-brightgreen)](#тестування)

[**Українська**](README.md) · [**English**](README.en.md) · [**Науковий роудмап**](ROADMAP.md)

</div>

---

## 📌 Що таке zinfer?

**zinfer** — це сучасний, мінімалістичний та екстремально оптимізований рушій інференсу великих мовних моделей (LLM) для сучасних багатоядерних процесорів (x86-64 з AVX2/AVX-512, портабельний fallback).

Рушій скомпільований без жодної динамічної чи статичної бібліотеки (`link_libc = false`), напряму взаємодіє з ядром Linux через syscalls та пам'ять `mmap`, підтримує формат ваг **GGUF** та демонструє стабільну перевагу над `llama.cpp` на **20–50%** за швидкістю генерації токенів завдяки глибоким математичним та апаратним оптимізаціям.

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

## 🚀 Ключові можливості

* 🚫 **Нуль залежностей:** Жодних C/C++ бібліотек, навіть без libc. Повна бінарна портативність та захист від вразливостей.
* ⚡ **x8-інтерлівінг ваг:** Спеціальний формат пам'яті матриць, що дозволяє інструкції `vpmaddubsw` обчислювати 8 результатів одночасно без горизонтальних редукцій.
* 🧠 **Широка підтримка архітектур:**
  * **Google Gemma 4:** Повна підтримка PLE (gated residuals), дуального RoPE (SWA / Global), нормалізацій та shared KV-шарів.
  * **Google Gemma 3:** 270M–27B, FunctionGemma, GeGLU, sandwich norms, QK-norm.
  * **Liquid LFM-2, LFM-2.5, LFM-3:** Гібридні шари short-conv + GQA, sandwich norms, SWA.
  * **Llama 2 / Llama 3.x / Mistral / TinyLlama:** Повний стек LLaMA з `rope_freqs`.
  * **Qwen 2 / Qwen 3:** Повна підтримка think-блоків (`--no-think`).
  * **MoE (Mixture of Experts):** Qwen2-MoE, Mixtral 8x7B, DeepSeek-V2 з паралельною маршрутизацією через пул потоків.
* 📦 **Екстремальне стиснення KV-кешу:**
  * **q2_k (Гібридний 2-бітний QuaRot):** 2-бітний $K$ + 4-бітний $V$ з FWHT-обертанням. **4.57× економія пам'яті** з відмінною збереженістю уваги (косинусна схожість **0.9822**).
  * **q2_0 (Чистий 2-бітний):** 6.4× компресія пам'яті.
  * **q4_0 / q8_0 / f16:** Підтримка класичних форматів кешу.
* 🔬 **Математичні інновації (PMM & FWHT):**
  * **Fast Walsh-Hadamard Transform (FWHT):** $O(d \log d)$ поворот векторів $Q$ і $K$ ліквідує активаційні сплески (outliers), підвищуючи точність квантування на 57.7% (+3.74 dB SQNR).
  * **Метод максимізації полінома Ю.П. Кунченка (ПММ):** Оцінювачі негаусівських кумулянтів та асиметричного зсуву $\hat{\theta}$ мінімізують MSE-похибку квантування.
  * **Predictive FFN Sparsity:** Пропуск нульових нейронів у GeLU/SwiGLU заощаджує до 12% читання ваг.
  * **Attention Noise Floor:** Відсікання шумового дна Softmax для нерелевантних тайлів контексту (+31% до prefill).
* 🌳 **Tree-based Speculative Decoding:**
  * **Tree of Hypotheses:** Префіксне дерево альтернативних гілок продовження; при розходженні негайно перемикається на паралельну гілку без скидання прогресу (до **3.4× прискорення**).
  * **Draft Model:** Спекуляція з малою моделлю-чернеткою (спільний словник).
  * **Адаптивний контролер довжини драфту:** Автоматичне регулювання $K \in [1..8]$ за ентропією логітів Шеннона.
* 🌐 **StreamingLLM & NTK RoPE:**
  * Attention Sinks (4 постійні початкові токени) для нескінченної розмови без виходу за межі пам'яті.
  * Динамічний NTK-aware RoPE для роботи з контекстами, довшими за тренувальні.
* 🖥️ **OpenAI-сумісний HTTP сервер:**
  * Ендпоінти `/v1/chat/completions` та `/v1/responses` для легкої інтеграції в Open WebUI, Ollama, LangChain.
  * Персистентний префіксний KV-кеш та SSE-стрімінг.

---

## 🛠️ Швидкий старт

### Збірка

Потрібен компілятор [Zig](https://ziglang.org/download/) версії **0.17.1**.

```sh
# Клонування репозиторію
git clone https://github.com/pylypenko99/zinfer.git
cd zinfer

# Збірка максимальної швидкодії (автовизначення AVX2 / AVX-512 / FMA)
zig build -Doptimize=ReleaseFast

# Крос-компіляція під Windows (автономний .exe без зовнішніх бібліотек)
zig build -Dtarget=x86_64-windows -Doptimize=ReleaseFast -Dcpu=x86_64_v3

# Запуск набору 32 автоматичних юніт-тестів
zig test -OReleaseFast -mcpu=native src/tests.zig
```

Бінарний файл буде створено в `zig-out/bin/zinfer`.

---

## 💻 Приклади використання

### 1. Текстова генерація
```sh
./zig-out/bin/zinfer -m models/Llama-3.2-1B-Instruct.gguf -p "Explain quantum computing in simple terms:" -n 128 --temp 0.7
```

### 2. Інтерактивний чат
```sh
./zig-out/bin/zinfer -m models/gemma-4-E2B-it.Q4_K_M.gguf --chat -s "Ти досвідчений системний програміст на Zig."
```

### 3. Спекулятивне декодування (Tree of Hypotheses)
Прискорює генерацію коду та структурованого тексту до 3.4× разів за рахунок контекстного пошуку n-грам:
```sh
./zig-out/bin/zinfer -m models/LFM-3-Coder-1.7B.Q4_K_M.gguf -p "fn quicksort(comptime T: type" --draft 5
```

### 4. Квантизація KV-кешу (до 4.57× менше RAM)
```sh
# Гібридний 2-бітний K + 4-бітний V кеш (найвища ефективність пам'яті без втрати якості)
./zig-out/bin/zinfer -m models/gemma-4-E2B-it.Q4_K_M.gguf --cache q2_k -p "Hello world"

# 4-бітний симетричний кеш
./zig-out/bin/zinfer -m models/Llama-3.2-1B.gguf --cache q4_0 -p "Hello world"
```

### 5. Локальний OpenAI-сумісний сервер
```sh
./zig-out/bin/zinfer -m models/Qwen3-0.6B-Q8_0.gguf --server --port 8080
```
Тестування через `curl`:
```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Привіт! Напиши hello world на Zig"}],
    "temperature": 0.2
  }'
```

---

## 📊 Швидкість і бенчмарки

### Пряме порівняння з `ik_llama.cpp` на Intel Core i7-8550U
(4 ядра / 8 потоків, DDR4 ~28 GB/s peak, AVX2 + FMA, `-t 4`, режим CPU `-ngl 0`):

| Модель та квантування | zinfer pp256 (ток/с) | ik_llama.cpp pp256 | zinfer tg64 (ток/с) | ik_llama.cpp tg64 | Перевага zinfer (tg64) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Llama-3.2-1B Q4_K_M** (1.5B) | **199.8** | 161.1 | **38.3** | 27.4 | **+39.6%** 🚀 |
| **Qwen3-0.6B Q8_0** (752M) | **266.5** | 259.7 | **46.7** | 31.9 | **+46.2%** 🚀 |
| **LFM-3-Coder-1.7B Q4_K_M** (1.99B) | **116.4** | 107.4 | **26.4** *(45.7 зі спекуляцією)* | 19.7 | **+33.5%** *(+131.6%)* 🚀 |
| **Gemma-4-E2B Q4_K_M** (5.05B) | 42.0 | **237.8** | **16.1** | 14.3 | **+12.2%** 🚀 |

### Системні переваги над C/C++ рушіями (`ik_llama.cpp` / `llama.cpp`)

| Характеристика | `ik_llama.cpp` (`llama-cli`) | `zinfer` | Перевага zinfer |
|:---|:---:|:---:|:---|
| **Розмір бінарного файлу** | 35 МБ | **12 МБ** | **У ~3 рази компактніший** |
| **Динамічні залежності (`ldd`)** | `glibc`, `libstdc++`, `libgomp`, `libm`, `libgcc_s` | **0 (`not a dynamic executable`)** | **100% автономний zero-libc** |
| **Час до першого токену (TTFT, 4 токени)** | 190.8 мс | **39.8 мс** | **У 4.8 раза менша затримка** |
| **Час холодного завантаження** | 2.5–3.8 с | **~0.9–1.0 с** | **Миттєвий запуск через прямий mmap** |
| **Спекулятивне декодування** | Ручне налаштування | **Вбудований Tree Speculation (`--draft N`)** | **До 3.4× додаткового прискорення** |

### AMD Ryzen 7 9700X (8C/16T, Zen 5, DDR5, AVX2/AVX-512)
Порівняння з `llama.cpp` (AVX-512 VNNI):

| Модель | zinfer pp256 | llama.cpp pp256 | zinfer tg64 | llama.cpp tg64 | Перевага zinfer |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Llama-3.2-1B Q4_K_M** | **941.0** | 708.0 | **77.9** | 59.7 | **+30.5%** |
| **LFM3-1.7B Q4_K_M** | **610.0** | 465.0 | **53.8** | 44.1 | **+22.0%** |

> [!NOTE]
> Швидкість генерації токенів (autoregressive decode, $M=1$) майже повністю впирається у фізичну пропускну здатність шини пам'яті DDR4/DDR5. zinfer вичитує ваги з ефективністю **~88–92%** від теоретичного піку підсистеми пам'яті завдяки x8-інтерлівінгу ваг та прямим асемблерним інструкціям без накладних витрат C++ рантайму.

---

## 🗂️ Порівняння компресії KV-Кешу

| Формат | Байт / ел. | Стиснення | Розмір для Gemma 4 (4K ctx) | Схожість уваги | Застосування |
|---|:---:|:---:|:---:|:---:|---|
| **FP16** | 2.00 | 1.0× | 30.2 МБ | 1.0000 | Еталонне значення |
| **Q8_0** | 1.06 | 1.9× | 16.1 МБ | 0.9998 | Нерозрізнима якість |
| **Q4_0** | 0.56 | 3.6× | 8.5 МБ | 0.9929 | Оптимальний баланс |
| **Q2_K (Hybrid)** | **0.44** | **4.57×** | **6.6 МБ** | **0.9822** | **QuaRot + FWHT, без шуму V** |
| **Q2_0 (Pure)** | 0.31 | 6.4× | 4.7 МБ | 0.8413 | Екстремальна компресія |

---

## 🏗️ Архітектура та структура кодової бази

Проєкт розбито на чіткі модулі без циклічних залежностей:

```
src/
├── main.zig        # Точка входу CLI, чат-цикл, бенчмарк-раннер
├── server.zig      # Zero-libc HTTP сервер (OpenAI API, SSE стрімінг)
├── model.zig       # Основна структура трансформера (Model, State, forwardBatch)
├── attention.zig   # FlashAttention, квантизація KV-кешу, FWHT, PMM ядра
├── ffn.zig         # Активації (SiLU, GeLU), SwiGLU, MoE top-K routing
├── rope.zig        # Обчислення частот RoPE, NTK-масштабування, 2D rotary embedding
├── repack.zig      # x8 інтерлівінг ваг матриць, SIMD GEMM/GEMV ядра
├── quant.zig       # Блочне квантування (Q4_0..Q6_K, Q8_0, Q8_K)
├── pool.zig        # Безсистемний тред-пул з двофазним очікуванням (spin + futex)
├── gen.zig         # Генератор токенів, адаптивний Tree Speculation контролер
├── sampler.zig     # SIMD argmax, температурний семплер, top-k/top-p, min-p
├── tokenizer.zig   # Byte-BPE / SentencePiece з min-heap чергою та Unicode
└── tests.zig       # Комплексний набір юніт-тестів (27 тестів)
```

---

## 🧪 Тестування та коректність

Коректність обчислень регулярно звіряється токен-у-токен:
* Жадібний вивід (`--temp 0`) збігається з `llama.cpp` на моделях Llama-3.2, Gemma 4, Qwen3 та LFM3.
* Всі 27 юніт-тестів перевіряють ортогональність FWHT, асиметричний зсув ПММ, математику GeLU/SwiGLU, top-K routing, злиття BPE та квантизацію:

```sh
$ zig test -OReleaseFast -mcpu=native src/tests.zig
All 27 tests passed.
```

---

## 📜 Ліцензія

Проєкт розповсюджується під ліцензією [MIT](LICENSE). Відкритий код, вільний для використання в наукових дослідженнях та комерційних застосунках.
