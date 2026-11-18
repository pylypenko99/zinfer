# Квантизатор GGUF/ggml на Zig — референс для zinfer (P2-A/P2-B/P2-C)

> Ця документація — довідник для додавання форматів ваг (Q2_K/Q3_K, IQ\*, TQ\*, MXFP4/NVFP4) у `quant.zig`/`repack.zig` zinfer. Написана як супутній план окремого інструменту-квантизатора (zquant), тому розділи про CLI/потоки — довідково; для zinfer головне: розд. 1–7 (формати, layout-и, ядра, LUT).

## 1. Цілі та пріоритети

| Пріоритет | Рішення |
|---|---|
| Швидкість | SIMD через `@Vector`, single-pass, `@setRuntimeSafety(false)`, ReleaseFast, mmap |
| Легкість | 0 залежностей, статичний бінарник ~100–300 КБ |
| Останні кванти | IQ1_M/IQ1_S, IQ2/IQ3/IQ4, TQ1_0/TQ2_0 (BitNet), MXFP4 (gpt-oss), NVFP4, Q1_0/Q2_0/F8_E4M3 |
| Малий розмір моделі | 1.125–2.5 bpw + importance matrix (imatrix) |
| Продуктивність | квантування memory-bound: метрика GB/s на ядро, ціль — пропускна здатність RAM |

## 2. Мінімум теорії

### 2.1 Блочне афінне квантування (legacy Q4_0…Q8_0)
Ваги діляться на блоки (32 елем.). Для блоку зберігається масштаб `d: f16` і цілі коди:
`x[i] ≈ d · (q[i] - 8)` (Q4_0, q∈[0,15]) або `x[i] ≈ d · q[i]` (Q8_0, q∈[-127,127]).
`d = max(x)/-8` для Q4_0, `d = max|x|/127` для Q8_0, `q = round(x·(1/d))` з clamp.

### 2.2 K-кванти (суперблоки по 256)
Дворівневі масштаби: 256 елем. = суперблок з `d, dmin: f16`, підблоки (16 або 32 елем.) мають власні масштаби, заквантовані в 4–8 біт через суперблоковий масштаб. Пошук оптимальних `(d, min)` — ітеративна оптимізація зваженої MSE.

### 2.3 I-кванти (IQ1/IQ2/IQ3/IQ4) — «екстремально малі»
Група з 8 (IQ2/IQ3) або 16 (IQ1) елементів кодується одним індексом у загальній таблиці (LUT/grid). Таблиці (`iq2xxs_grid[256]`, `iq2xs_grid[512]`, `iq2s_grid[1024]`, `iq3xxs_grid[256]`, `iq1s_grid`, `kvalues_iq4nl`…) генеруються k-means offline і просто копіюються з `ggml-common.h` (ліцензія MIT). Пошук: для кожної групи перебрати всі коди LUT, взяти argmin зваженої похибки. Обовʼязково з imatrix.

### 2.4 Найновіші формати
- **TQ1_0 / TQ2_0** — тернарні {-1,0,+1} для BitNet-подібних моделей. TQ1_0 пакує 5 тритів у байт (3⁵=243<256).
- **MXFP4** — формат gpt-oss: блок 32, E2M1-значення + масштаб E8M0 (u8) на блок.
- **NVFP4** — блок 64, E2M1 + 4 підмасштаби UE4M3 (E4M3) по 16 елем.
- **Q1_0 / Q2_0 / F8_E4M3** — нові «справжні» 1-біт (1.125 bpw), 2-біт (2.25 bpw) та float8.

## 3. Каталог форматів (бит-в-біт по ggml-common.h + ggml.h master)

| Тип | `ggml_type` ID | елем./блок | байт/блок | bpw | Призначення |
|---|---|---|---|---|---|
| Q4_0 | 2 | 32 | 18 | 4.50 | легкий legacy |
| Q4_1 | 3 | 32 | 20 | 5.00 | legacy з min |
| Q5_0 | 6 | 32 | 22 | 5.50 | legacy |
| Q5_1 | 7 | 32 | 24 | 6.00 | legacy |
| Q8_0 | 8 | 32 | 36 | 8.50 | ref/проміжний |
| Q2_K | 10 | 256 | 84 | 2.625 | K-кванти |
| Q3_K | 11 | 256 | 110 | 3.4375 | K-кванти |
| Q4_K | 12 | 256 | 144 | 4.50 | K-кванти |
| Q5_K | 13 | 256 | 176 | 5.50 | K-кванти |
| Q6_K | 14 | 256 | 210 | 6.5625 | макс. якість K |
| IQ2_XXS | 16 | 256 | 66 | 2.0625 | екстремально малий |
| IQ2_XS | 17 | 256 | 74 | 2.3125 | екстремально малий |
| IQ3_XXS | 18 | 256 | 98 | 3.0625 | 3 біти |
| IQ1_S | 19 | 256 | 50 | 1.5625 | ~1.5 біт |
| IQ4_NL | 20 | 32 | 18 | 4.50 | нелінійний 4-біт |
| IQ3_S | 21 | 256 | 110 | 3.4375 | 3 біти |
| IQ2_S | 22 | 256 | 82 | 2.5625 | 2.5 біти |
| IQ4_XS | 23 | 256 | 136 | 4.25 | 4 біти best |
| IQ1_M | 29 | 256 | 56 | 1.75 | ~1.75 біт |
| BF16 | 30 | — | — | 16 | gpt-oss та ін. |
| **TQ1_0** | 34 | 256 | 54 | 1.6875 | тернарний (BitNet) |
| **TQ2_0** | 35 | 256 | 66 | 2.0625 | тернарний |
| **MXFP4** | 39 | 32 | 17 | 4.25 | gpt-oss нативно |
| **NVFP4** | 40 | 64 | 36 | 4.5 | NV-стиль FP4 |
| **Q1_0** | 41 | 128 | 18 | **1.125** | новітній 1-біт |
| **Q2_0** | 42 | 64 | 18 | 2.25 | новітній 2-біт |
| F8_E4M3 | 43 | — | — | 8 | float8 |

> ⚠️ `enum ggml_type` у ggml додається тільки в кінець, тому ID стабільні. У GGUF писати `general.quantization_version` (зараз `GGML_QNT_VERSION = 2`).

## 4. Визначення блоків у Zig (з гарантією ABI)

```zig
// blocks (у zinfer — інтегрувати в quant.zig)
const std = @import("std");

pub const QK_K: usize = 256;

pub const BlockQ8_0 = extern struct { d: f16, qs: [32]i8 };
pub const BlockQ4_0 = extern struct { d: f16, qs: [16]u8 };
pub const BlockQ2_K = extern struct {   // 84 байти
    scales: [QK_K/16]u8,   // scale+min по 4 біти
    qs: [QK_K/4]u8,
    d: f16, dmin: f16,
};
pub const BlockQ6_K = extern struct {
    ql: [QK_K/2]u8, qh: [QK_K/4]u8, scales: [QK_K/16]i8, d: f16,
};
pub const BlockIQ2XXS = extern struct { d: f16, qs: [QK_K/8]u16 };   // 32 групи по 8 елем., індекс у grid256
pub const BlockTQ2_0 = extern struct { qs: [QK_K/4]u8, d: f16 };
pub const BlockMXFP4 = extern struct { e: u8, qs: [16]u8 };           // E8M0 масштаб + E2M1 по 2 на байт

comptime {
    if (@sizeOf(BlockQ8_0) != 36) @compileError("BlockQ8_0 != 36");
    if (@sizeOf(BlockQ4_0) != 18) @compileError("BlockQ4_0 != 18");
    if (@sizeOf(BlockQ2_K) != 84) @compileError("BlockQ2_K != 84");
    if (@sizeOf(BlockQ6_K) != 210) @compileError("BlockQ6_K != 210");
    if (@sizeOf(BlockIQ2XXS) != 66) @compileError("BlockIQ2XXS != 66");
    if (@sizeOf(BlockTQ2_0) != 66) @compileError("BlockTQ2_0 != 66");
    if (@sizeOf(BlockMXFP4) != 17) @compileError("BlockMXFP4 != 17");
}
```

**Правило:** layout біт-в-біт як у `ggml-common.h`; `extern struct` гарантує C-layout; перевірки `@sizeOf` — обовʼязково.

## 5. Ядро квантування

### 5.1 Q8_0 — еталон з SIMD

```zig
const V8 = @Vector(8, f32);

pub fn quantRowQ8_0(x: []const f32, y: []BlockQ8_0) void {
    @setRuntimeSafety(false);
    for (y, 0..) |*blk, bi| {
        const xr = x[bi*32..][0..32];
        var amax: f32 = 0;
        comptime var j: usize = 0;
        inline while (j < 32) : (j += 8) {
            const v: V8 = @bitCast(xr[j..][0..8].*);
            amax = @max(amax, @reduce(.Max, @abs(v)));
        }
        const d: f32 = amax / 127.0;
        const id: V8 = @splat(if (d != 0) 1.0 / d else 0.0);
        blk.d = @floatCast(d);
        comptime var k: usize = 0;
        inline while (k < 32) : (k += 8) {
            const v: V8 = @bitCast(xr[k..][0..8].*);
            const qc = @round(v * id);
            const cl = @min(@max(qc, @as(V8, @splat(-127.0))), @as(V8, @splat(127.0)));
            blk.qs[k..][0..8].* = @bitCast(@as(@Vector(8, i8), @intFromFloat(cl)));
        }
    }
}
```

Ключові Zig-механіки: `@bitCast` масив↔вектор, `@reduce(.Max)`, `inline while` з comptime-лічильником = ручний unroll, векторні `@min/@max/@round/@intFromFloat`.

### 5.2 Q4_0 — бітпакінг nibble

```zig
pub fn quantRowQ4_0(x: []const f32, y: []BlockQ4_0) void {
    @setRuntimeSafety(false);
    for (y, 0..) |*blk, bi| {
        const xr = x[bi*32..][0..32];
        var amax: f32 = 0; var vmax: f32 = 0;
        inline for (0..4) |c| {
            const v: V8 = @bitCast(xr[c*8..][0..8].*);
            amax = @max(amax, @reduce(.Max, @abs(v)));
            vmax = @max(vmax, @reduce(.Max, v));   // знаковий max!
        }
        const d: f32 = vmax / -8.0;
        const id: f32 = if (d != 0) 1.0 / d else 0.0;
        blk.d = @floatCast(d);
        var q: [32]u8 = undefined;
        for (0..32) |i|
            q[i] = @intFromFloat(@min(@max(@round(xr[i]*id) + 8.0, 0.0), 15.0));
        for (0..16) |i| blk.qs[i] = q[i] | (q[i+16] << 4);   // [j]=lo, [j+16]=hi
    }
}
```

### 5.3 K-кванти — дворівневі масштаби + ітераційна оптимізація

Схема Q4_K (суперблок 256 = 8 підблоків по 32):

```
d, dmin: f16          ← масштаби суперблока
scales[12]: u8        ← 8×(6 біт scale + 6 біт min)
qs[128]: u8           ← nibbles
x ≈ d·(scale_6bit/64)·q + dmin·(min_6bit/64),  q ∈ [0,15]
```

Ядро пошуку оптимального масштабу підблоку (мінімізація зваженої MSE; `w` — imatrix-ваги або 1):

```zig
fn refineDMin(x: []const f32, w: []const f32, q: []const i8,
              d0: f32, min0: f32, n_iter: usize) struct { d: f32, min: f32 } {
    var d = d0; var m = min0;
    for (0..n_iter) |_| {
        var a: f64 = 0; var b: f64 = 0; var c: f64 = 0;
        var d1: f64 = 0; var e1: f64 = 0;
        for (x, w, q) |xi, wi, qi| {
            const qf: f64 = @floatFromInt(qi);
            const e: f64 = xi - d*qf - m;
            const wf: f64 = wi;
            a += wf*qf*qf; b += wf*qf; c += wf;
            d1 += wf*qf*e; e1 += wf*e;
        }
        const det = a*c - b*b;
        if (@abs(det) < 1e-12) break;
        const dd = ( d1*c - b*e1) / det;
        const dm = (-b*d1 + a*e1) / det;
        d += @floatCast(dd); m += @floatCast(dm);
    }
    return .{ .d = d, .min = m };
}
```

Повна процедура Q4_K: (1) референсне квантування в int8 (Q8_K), (2) стартові `d/min` з min/max, (3) `refineDMin` → requant, 3–5 ітерацій з раннім виходом, (4) пакування масштабів у 6 біт через суперблокові `d/dmin` (дзеркалить `make_qkx2_quants` у ggml-quants.c). Для біт-в-біт сумісності з llama.cpp — точний порядок float-операцій reference-коду.

### 5.4 I-кванти — LUT-пошук

```zig
// Grid розпаковано один раз у f16-таблицю: grid_u[256][8]
pub fn quantGroupIQ2XXS(
    x: []const f32, w: []const f32, scale: f32,
    grid_u: *const [256][8]f32,
) struct { code: u8, err: f64 } {
    var best: u8 = 0; var best_err: f64 = std.math.floatMax(f64);
    for (grid_u, 0..) |g, c| {
        var err: f64 = 0;
        inline for (0..8) |i| {
            const e = @as(f64, w[i]) * (@as(f64, x[i]) - @as(f64, scale) * g[i]);
            err += e * e;
        }
        if (err < best_err) { best_err = err; best = @intCast(c); }
    }
    return .{ .code = best, .err = best_err };
}
```

Повний цикл i-кванту (як `quantize_iq2_xxs` у llama.cpp): на суперблок (256) — кілька спроб масштабу `d` (старт з max|w·x|, потім множники ×1.05, ×0.95…), кожна спроба = вибір найкращого коду на групу + зважена MSE; тримаємо найкращий `d`. IQ1_M/IQ1_S додають per-16 delta-грід і 3-бітні підмасштаби.

Таблиці `iq2xxs_grid`, `ksigns_iq2xs`, `kmask_iq2xs`, `kvalues_iq4nl` тощо — конвертувати з `ggml-common.h` (~10 КБ даних, `align(64)` на кожну таблицю). Для **деkвантизації в zinfer** (dot-ядра) гріди потрібні ті самі.

### 5.5 Тернарні та FP4

- **TQ2_0**: `q = sign(x)·(|x|>τ)`, 2 біти/елем., `d = mean|x|` (пошук порогу τ по MSE).
- **TQ1_0**: те саме + п'ять тритів у байт: `code = Σ tᵢ·3^i` (0…242).
- **MXFP4**: E2M1 ∈ {0,±0.5,±1,±1.5,±2,±3,±4,±6}; масштаб E8M0 = `2^round(log2(max/6))`.
- **NVFP4**: як MXFP4, але масштаб E4M3 на 16 елем. (4 на блок 64).
- **Q1_0** (1.125 bpw): sign-bit на елемент + масштаб f16 на 128.

## 6. SIMD-стратегія

1. Ширина: 8×f32 для AVX2/SSE, 16×f32 для AVX-512, 4×f32 NEON.
2. `@bitCast` масив↔вектор — найшвидший load/store.
3. Редукції — `@reduce(.Add, v)`; для біт-сумісності повторювати порядок додавань reference-коду.
4. FMA: `@mulAdd(f32, a, b, c)`.
5. Branchless: `@select(u8, cond, a, b)`.
6. LUT у кеші: grids ≤16 КБ — `align(64)`, не копіювати в heap.

## 7. Тестування та валідація

1. **Comptime**: `@sizeOf` усіх блоків.
2. **Round-trip**: quant→dequant на синтетиці — RMSE ~`d/√12`, косинус >0.999 (Q4+), >0.97 (IQ2).
3. **Golden-тести**: порівняння з `llama-quantize` (байти або dequant з допуском).
4. **Fuzz**: `std.Random` з сидами, safety увімкнена в тестах.
5. **Якість**: KL/RMSE на тензор + perplexity через `llama-perplexity`.
6. **Еталонні моделі**: Qwen3-0.6B (BF16 вхід), SmolLM2-360M (F16 вхід) — повний workflow golden/ppl/розміри.

## 8. Джерела

- ggml.h (llama.cpp master) — enum ggml_type: TQ1_0=34, TQ2_0=35, MXFP4=39, NVFP4=40, Q1_0=41, Q2_0=42, F8_E4M3=43, QNT_VERSION=2 — https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h
- ggml-common.h (ggml master) — layout-и блоків, QK_K=256, grids/LUT — https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h
- DeepWiki: Quantization Techniques (llama.cpp) — https://deepwiki.com/ggml-org/llama.cpp/7.3-quantization-techniques
