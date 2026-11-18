# Технічний план реалізації Low-RAM інференсу для MoE-моделей у zinfer (Вивантаження на диск + Потоковий стрімінг експертів під час обчислень)

Цей документ є готовою специфікацією та покроковим планом для розробника або ШІ-агента з реалізації динамічного вивантаження неактивних експертів (Disk Eviction) та **асинхронного стрімінгу потрібних експертів з диска прямо під час обчислень** для **Mixture-of-Experts (MoE)** моделей (`qwen2moe`, `mixtral`, `deepseek2`, `qwen3moe`) у рушії [pylypenko99/zinfer](https://github.com/pylypenko99/zinfer).

## 1. Головний принцип: Розподіл MoE-моделі між RAM і NVMe SSD

У MoE-архітектурах, які підтримує [pylypenko99/zinfer](https://github.com/pylypenko99/zinfer) у файлах [src/model.zig](https://github.com/pylypenko99/zinfer/blob/main/src/model.zig) та [src/ffn.zig](https://www.google.com/search?q=https://github.com/pylypenko99/zinfer/blob/main/src/ffn.zig), ваги поділяються на постійно активне ядро (у RAM) та масив експертів (на диску з гарячим кешем у RAM):

| Компонент MoE-моделі | Частка від розміру файлу | Частота використання | Де знаходиться при малому RAM | Модуль у `zinfer` |
| :--- | :--- | :--- | :--- | :--- |
| **Шари уваги (`wq`, `wk`, `wv`, `wo`), RMSNorm, Ембедінги** | **8–12%** | **100%** (на кожному токені) | Постійно закріплені в RAM (`Pinned` / `mlock`) | [src/attention.zig](https://github.com/pylypenko99/zinfer/blob/main/src/attention.zig), [src/model.zig](https://github.com/pylypenko99/zinfer/blob/main/src/model.zig) |
| **Шлюзи роутерів (`ffn_gate_inp`) та Shared Experts** | **1–3%** (роутери — десятки КБ) | **100%** (визначають маршрут на кожному кроці) | Постійно закріплені в RAM (`Pinned`) | [src/ffn.zig](https://www.google.com/search?q=https://github.com/pylypenko99/zinfer/blob/main/src/ffn.zig), [src/model.zig](https://github.com/pylypenko99/zinfer/blob/main/src/model.zig) |
| **KV-кеш контексту (`--cache q2_k`)** | Фіксовано (десятки МБ) | **100%** (на кожному токені) | Постійно в RAM у форматі `q2_k` (стиснення у 4.57 раза) | [src/attention.zig](https://github.com/pylypenko99/zinfer/blob/main/src/attention.zig) |
| **Маршрутизовані експерти (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`)** | **85–90%** усього обсягу моделі | Лише **Top-K** (4 з 64 або 8 зі 128 — **6–12%** на токен) | **Холодні — на диску**; **гарячі — у RAM-кеші**; **відсутні — стрімляться у фоні під час обчислень** | [src/ffn.zig](https://www.google.com/search?q=https://github.com/pylypenko99/zinfer/blob/main/src/ffn.zig), [src/repack.zig](https://github.com/pylypenko99/zinfer/blob/main/src/repack.zig), [src/platform.zig](https://github.com/pylypenko99/zinfer/blob/main/src/platform.zig) |

## 2. Як «вивантажувати» експертів на диск за 0 мс без запису на SSD

У [pylypenko99/zinfer](https://github.com/pylypenko99/zinfer) модуль [src/repack.zig](https://github.com/pylypenko99/zinfer/blob/main/src/repack.zig) за замовчуванням виділяє анонімну оперативну пам'ять через `allocAnonymous` і переупаковує квантовані матриці в `x8`-формат.
* Якщо намагатися вивантажити анонімну пам'ять (`allocAnonymous`), ОС змушена **фізично записувати (write)** дані у swap-файл, що гальмує інференс і зношує SSD.
* Якщо ж зберегти `x8`-переупаковані матриці експертів у файл кешу на диску (`<model>.x8cache`) і відкрити його через `platform.mapFileReadOnly` у [src/platform.zig](https://github.com/pylypenko99/zinfer/blob/main/src/platform.zig) (плюс тримати відкритим файловий дескриптор `fd` для прямого асинхронного `pread`), то сторінки експертів у RAM залишаються **чистими файловими сторінками (clean file-backed pages)**.
* Вивантаження непотрібного експерта на диск виконується одним викликом `madvise(expert_slice, MADV_DONTNEED)` (або звільненням слота в кільцевому буфері стрімінгу): ядро ОС **миттєво і без жодного запису на диск** звільняє фізичні сторінки RAM.

## 3. Архітектура стрімінгу експертів прямо під час обробки (Zero-Stall Expert Streaming)

Щоб процесор не простоював у очікуванні диска, коли роутер обирає експерта, якого зараз немає в RAM, стрімінг у [pylypenko99/zinfer](https://github.com/pylypenko99/zinfer) організовується у **три паралельні ешелони перекриття (Compute + I/O Overlap)**:

| Ешелон перекриття | Коли стартує читання з диска | На фоні яких обчислень CPU завантажується експерт | Де відбувається синхронізація |
| :--- | :--- | :--- | :--- |
| **1. Міжшаровий (Pre-Attention Lookahead)** | На самому початку шару `l` (до запуску QKV та FlashAttention) | Поки [src/attention.zig](https://github.com/pylypenko99/zinfer/blob/main/src/attention.zig) рахує QKV-проєкції, RoPE, `fwhtSimd`, `FlashDecoding` (`kv_tile = 16`) та `wo` | Перед входом у `moeFfn` у [src/ffn.zig](https://www.google.com/search?q=https://github.com/pylypenko99/zinfer/blob/main/src/ffn.zig) |
| **2. Внутрішньошаровий (`Hits`-First Scheduling)** | Одразу після точного обчислення роутера `ffn_gate_inp` шару `l` | Поки пул потоків [src/pool.zig](https://www.google.com/search?q=https://github.com/pylypenko99/zinfer/blob/main/src/pool.zig) рахує `shared_experts` та тих із Top-K експертів, які **вже є в RAM (`Hits`)** | По мірі готовності кожного окремого експерта зі списку `Misses` |
| **3. Поматричний (Sub-Expert Pipelining: `gate/up` -> `down`)** | Читання експерта розбите на 2 частини: спочатку `gate + up`, потім `down` | Поки SIMD-потоки рахують `gate(x)`, `up(x)` та нелінійність `SwiGLU`, з диска дочитується матриця `down` цього ж експерта | Атомарний стан слота `stage: Atomic(u32)` (`0` = empty, `1` = gate_up_ready, `2` = down_ready) |

## 4. Покроковий план модифікації коду `zinfer` (ТЗ для ШІ-моделі)

Усі зміни виконуються на чистому **Zig 0.17.1** (`link_libc = false`) із використанням наявного стилю синхронізації на атомарних змінних і `futex` із [src/pool.zig](https://www.google.com/search?q=https://github.com/pylypenko99/zinfer/blob/main/src/pool.zig).

### Крок 1: Додати системні виклики підкачки та прямого читання у `src/platform.zig`

1. Відкрити [src/platform.zig](https://github.com/pylypenko99/zinfer/blob/main/src/platform.zig) і додати управління сторінками `mmap` та неблокуюче позиційне читання:
   ```zig
   pub const PageAdvice = enum {
       will_need, // Асинхронний readahead у сторінки mmap з диска
       dont_need, // Миттєве звільнення фізичного RAM (вивантаження на диск)
       random,    // Вимкнення зайвого readahead сусідніх неактивних експертів
   };

   pub fn advisePages(slice: []const u8, advice: PageAdvice) void {
       if (slice.len == 0) return;
       const page_size: usize = 4096;
       const addr = @intFromPtr(slice.ptr);
       const aligned_addr = addr & ~(page_size - 1);
       const aligned_len = ((addr + slice.len + page_size - 1) & ~(page_size - 1)) - aligned_addr;
       const ptr: [*]align(4096) u8 = @ptrFromInt(aligned_addr);
       // На Linux/macOS: std.posix.madvise(ptr[0..aligned_len], flag)
       // На Windows: PrefetchVirtualMemory / DiscardVirtualMemory
       _ = ptr;
       _ = aligned_len;
       _ = advice;
   }

   /// Пряме позиційне читання з файлу моделі/кешу у виділений стрімінг-слот RAM
   pub fn preadExact(fd: std.posix.fd_t, dst: []u8, offset: u64) void {
       var read_total: usize = 0;
       while (read_total < dst.len) {
           const n = std.posix.pread(fd, dst[read_total..], offset + read_total) catch break;
           if (n == 0) break;
           read_total += n;
       }
   }
   ```
2. У [src/gguf.zig](https://github.com/pylypenko99/zinfer/blob/main/src/gguf.zig) у функції `File.open` зберігати відкритий файловий дескриптор `fd` у структурі `File` (замість негайного закриття після `mmap`), щоб фоновий I/O-потік міг паралельно виконувати `preadExact` або `advisePages`.

### Крок 2: Поекспертне вирівнювання у `src/repack.zig`

1. У [src/repack.zig](https://github.com/pylypenko99/zinfer/blob/main/src/repack.zig) для тензорів експертів (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) забезпечити поекспертну розкладку у файлі `<model>.x8cache`:
   * Для кожного шару `l` і кожного експерта `e` матриці `gate` та `up` розташовуються одним суцільним блоком (`gate_up_chunk`), а матриця `down` — одразу за ними (`down_chunk`), причому початок кожного експерта вирівнюється до `4096` байтів.
   * Таке групування дозволяє зчитати `gate + up` за одну лінійну операцію читання з NVMe і негайно почати обчислення експерта, поки дочитується `down`.

### Крок 3: Фоновий I/O-стрімер та кільцевий буфер слотів (`ExpertStreamer`) у `src/ffn.zig`

Додати у [src/ffn.zig](https://www.google.com/search?q=https://github.com/pylypenko99/zinfer/blob/main/src/ffn.zig) (або окремий модуль `src/streamer.zig`, підключений до [src/ffn.zig](https://www.google.com/search?q=https://github.com/pylypenko99/zinfer/blob/main/src/ffn.zig)) фоновий потік підкачки з двофазною синхронізацією початку (`gate+up`) і кінця (`down`) завантаження експерта:

```zig
const std = @import("std");

pub const StreamStage = enum(u32) {
    empty = 0,
    loading = 1,
    gate_up_ready = 2, // Можна запускати GEMV/GEMM для gate та up + SwiGLU!
    down_ready = 3,    // Можна запускати GEMV/GEMM для down!
};

pub const StreamSlot = struct {
    layer: usize = 0,
    expert_id: usize = 0,
    stage: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    gate_up_buf: []align(64) u8, // Буфер під переупаковані x8 ваги gate + up
    down_buf: []align(64) u8,    // Буфер під переупаковані x8 ваги down
};

pub const ExpertStreamer = struct {
    fd: std.posix.fd_t,
    slots: []StreamSlot, // Кільцевий буфер на 4..8 транзитних слотів у RAM
    queue_head: std.atomic.Value(u32),
    queue_tail: std.atomic.Value(u32),
    io_thread: std.Thread,

    /// Викликається I/O-потоком: читає експерта з диска у дві фази
    fn ioWorkerLoop(self: *ExpertStreamer) void {
        while (true) {
            // 1. Очікує нове завдання у черзі (spin + futex аналогічно src/pool.zig)
            // 2. Фаза А: читає (або робить madvise .will_need + repack) матриці gate та up у slot.gate_up_buf
            // 3. Публікує готовність першої половини експерта:
            //    slot.stage.store(@intFromEnum(StreamStage.gate_up_ready), .release);
            //    std.Thread.Futex.wake(&slot.stage, 1);
            // 4. Фаза Б: одразу читає матрицю down у slot.down_buf, поки CPU вже рахує gate та up!
            // 5. Публікує повну готовність експерта:
            //    slot.stage.store(@intFromEnum(StreamStage.down_ready), .release);
            //    std.Thread.Futex.wake(&slot.stage, 1);
            _ = self;
            break;
        }
    }
};
```

### Крок 4: Конвеєрне виконання MoE-шару (`Hits`-First + Live Stream Consumption) у `src/ffn.zig`

Переписати функцію обчислення MoE у [src/ffn.zig](https://www.google.com/search?q=https://github.com/pylypenko99/zinfer/blob/main/src/ffn.zig) так, щоб вона ніколи не блокувалася на початку шару, а обробляла експертів у порядку їхньої готовності:

1. **Розділення Top-K на `Hits` та `Misses` відразу після роутера**:
   * Роутер обчислює `top_k` експертів `[e_1, e_2, ..., e_k]` та їхні ваги `weights[1..k]`.
   * Проходимо по списку `top_k`:
     * Якщо експерт `e_i` вже знаходиться у постійному чи гарячому RAM-кеші (`in_ram == true`), додаємо його у локальний масив `ram_hits`.
     * Якщо експерта `e_i` немає в RAM (`!in_ram`), **миттєво ставимо його в чергу `ExpertStreamer`** (`enqueueLoad(layer, e_i)`), прив'язуючи до вільного `StreamSlot`, і додаємо індекс слота у масив `disk_misses`.
2. **Етап А — Обчислення того, що вже в RAM (поки диск читає `disk_misses`)**:
   * Якщо шар має спільні експерти (`shared_experts` у `deepseek2` чи `qwen2moe`), пул потоків [src/pool.zig](https://www.google.com/search?q=https://github.com/pylypenko99/zinfer/blob/main/src/pool.zig) обчислює їх першими.
   * Далі пул потоків по черзі обчислює всі експерти зі списку `ram_hits` (оскільки їхні ваги вже в RAM, SIMD-ядра працюють на максимальній швидкості, даючи NVMe SSD форові мілісекунди на підкачку `disk_misses`).
3. **Етап Б — Потокове споживання експертів з диска по частинах (`Sub-Expert Execution`)**:
   * Для кожного слота `slot` зі списку `disk_misses`:
     * **Очікування Фази 1 (`gate_up_ready`)**: Потік перевіряє `slot.stage.load(.acquire)`. Якщо значення менше за `gate_up_ready`, чекає через `Futex.wait`. Щойно `stage >= gate_up_ready`:
     * Пул потоків [src/pool.zig](https://www.google.com/search?q=https://github.com/pylypenko99/zinfer/blob/main/src/pool.zig) **негайно запускає паралельне множення `gate` та `up` і активацію `SwiGLU`** з `slot.gate_up_buf`, не чекаючи завершення читання матриці `down` з диска!
     * **Очікування Фази 2 (`down_ready`)**: Після завершення `SwiGLU` перевіряється `slot.stage.load(.acquire) == down_ready` (на цей момент I/O-потік вже закінчив читати `down_buf` паралельно з обчисленням `gate + up`).
     * Запускається множення `down` з `slot.down_buf` і результат додається у вихідний акумулятор з вагою роутера `weights[i]`.
4. **Етап В — Просування в гарячий кеш або миттєве вивантаження**:
   * Після додавання результату експерта у вихідний вектор оновлюємо його `freq_score` у `MoECacheManager`:
     * Якщо цей експерт став «гарячим» (його `freq_score` перевищив мінімальний рейтинг резидентних експертів у RAM), копіюємо/переміщуємо його у постійний RAM-кеш, а найхолоднішого резидента вивантажуємо через `platform.advisePages(victim_slice, .dont_need)`.
     * Якщо ж це рідкісний («транзитний») експерт, просто скидаємо `slot.stage.store(0, .release)` (та викликаємо `.dont_need` для його діапазону у `mmap`), миттєво звільняючи слот під наступний шар!

### Крок 5: Міжшаровий старт стрімінгу ще до Attention у `src/model.zig`

Щоб почати завантаження експертів шару `l` ще раніше — до того, як завершиться блок уваги шару `l`:

1. У функції `forward` у [src/model.zig](https://github.com/pylypenko99/zinfer/blob/main/src/model.zig) перед викликом уваги шару `l` (коли вхідний вектор шару `x_in` вже відомий) виконати швидку оцінку маршруту:
   * Оскільки в MoE-трансформерах вхід блоку FFN дорівнює `x_mid = x_in + Attention(RMSNorm(x_in))`, вектор `x_in` вже містить основну компоненту напрямку для роутера `layer[l].ffn_gate_inp`.
   * Обчислити попередній `top_k_preview` множенням нормованого `x_in` на `layer[l].ffn_gate_inp` (це займає менше ніж 5 мікросекунд, бо матриця роутера крихітна і лежить у RAM).
   * Для тих експертів із `top_k_preview`, яких немає в RAM, одразу викликати `platform.advisePages(expertSlice(l, e), .will_need)` (або поставити пріоритетний pre-load у `ExpertStreamer`).
2. Поки процесор виконує важкі операції `wq`, `wk`, `wv`, `FlashDecoding` ([src/attention.zig](https://github.com/pylypenko99/zinfer/blob/main/src/attention.zig)) та `wo` на шарі `l`, NVMe SSD вже прокачує потрібних експертів цього шару в оперативну пам'ять.
3. Коли керування доходить до реального роутера перед FFN (де рахується точний `x_mid`), 80–90% потрібних з диска експертів вже знаходяться у стані `gate_up_ready` або `down_ready`.

## 5. План верифікації та тестування

1. **Перевірка побітової точності стрімінгу**: У [src/tests.zig](https://www.google.com/search?q=https://github.com/pylypenko99/zinfer/blob/main/src/tests.zig) створити тест для MoE-шару з лімітом слотів `max_resident_experts = 2` при `top_k = 4`, щоб гарантовано спрацьовували обидві гілки (`ram_hits` та двофазний стрімінг `disk_misses` через `gate_up_ready` -> `down_ready`), і звірити вихідний тензор із еталонним прогоном у повному RAM.
2. **Профілювання часу очікування диска (`io_wait_ns`)**: Додати лічильник часу, проведеного у `Futex.wait` на `slot.stage`. При увімкненому Pre-Attention Lookahead та `Hits`-First плануванні частка часу очікування диска має складати менше ніж 10–15% від загального часу виконання шару на NVMe SSD.
