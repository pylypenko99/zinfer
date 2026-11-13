//! Block formats, activation quantization and dot-product kernels.
//!
//! Weights stay in their GGUF block format. Activations are quantized once per
//! matmul to 8-bit (Q8K for K-quants, Q8B for 32-element formats) so the inner
//! loops are integer multiply-adds: vpmaddubsw/vpmaddwd via LLVM intrinsics on
//! AVX2 (LLVM can still fold loads and schedule them), portable vectors elsewhere.
const std = @import("std");
const builtin = @import("builtin");
const GgmlType = @import("gguf.zig").GgmlType;

pub const has_avx2 = builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);

extern fn @"llvm.x86.avx2.pmadd.ub.sw"(@Vector(32, u8), @Vector(32, i8)) @Vector(16, i16);
extern fn @"llvm.x86.avx2.pmadd.wd"(@Vector(16, i16), @Vector(16, i16)) @Vector(8, i32);

pub const BlockQ4_0 = extern struct { d: f16, qs: [16]u8 };
pub const BlockQ8_0 = extern struct { d: f16, qs: [32]i8 };
pub const BlockQ4_K = extern struct { d: f16, dmin: f16, scales: [12]u8, qs: [128]u8 };
pub const BlockQ5_K = extern struct { d: f16, dmin: f16, scales: [12]u8, qh: [32]u8, qs: [128]u8 };
pub const BlockQ6_K = extern struct { ql: [128]u8, qh: [64]u8, scales: [16]i8, d: f16 };

/// Activation block for K-quant weights: 256 values, bsums are sums of 16.
/// qs comes first and the block is 32-aligned so vector loads never split.
pub const Q8K = extern struct { qs: [256]i8 align(32), d: f32, bsums: [16]i16, bsums32: [8]i16 };
/// Activation block for 32-element weight formats; s = d * sum(qs).
pub const Q8B = extern struct { qs: [32]i8 align(32), d: f32, s: f32 };

comptime {
    std.debug.assert(@sizeOf(BlockQ4_0) == 18);
    std.debug.assert(@sizeOf(BlockQ8_0) == 34);
    std.debug.assert(@sizeOf(BlockQ4_K) == 144);
    std.debug.assert(@sizeOf(BlockQ5_K) == 176);
    std.debug.assert(@sizeOf(BlockQ6_K) == 210);
}

pub const ActKind = enum { f32, q8k, q8b };

pub fn actKind(t: GgmlType) ActKind {
    return switch (t) {
        .q4_k, .q5_k, .q6_k => .q8k,
        .q4_0, .q8_0 => .q8b,
        else => .f32,
    };
}

pub fn isSupported(t: GgmlType) bool {
    return switch (t) {
        .f32, .f16, .bf16, .q4_0, .q8_0, .q4_k, .q5_k, .q6_k => true,
        else => false,
    };
}

// ---------------------------------------------------------------------------
// SIMD helpers

pub const V32u8 = @Vector(32, u8);
pub const V32i8 = @Vector(32, i8);
pub const V16i16 = @Vector(16, i16);
pub const V8i32 = @Vector(8, i32);
pub const V8f = @Vector(8, f32);
pub const V16f = @Vector(16, f32);

fn evenOdd(comptime n: usize, comptime odd: bool) [n]i32 {
    var m: [n]i32 = undefined;
    for (&m, 0..) |*e, i| e.* = @intCast(2 * i + @intFromBool(odd));
    return m;
}

const even16 = evenOdd(16, false);
const odd16 = evenOdd(16, true);
const even8 = evenOdd(8, false);
const odd8 = evenOdd(8, true);

/// u8 x i8 -> pairwise-summed i16 with saturation (vpmaddubsw).
pub inline fn maddubs(a: V32u8, b: V32i8) V16i16 {
    if (has_avx2) return @"llvm.x86.avx2.pmadd.ub.sw"(a, b);
    const ae: @Vector(16, i32) = @shuffle(u8, a, undefined, even16);
    const ao: @Vector(16, i32) = @shuffle(u8, a, undefined, odd16);
    const be: @Vector(16, i32) = @shuffle(i8, b, undefined, even16);
    const bo: @Vector(16, i32) = @shuffle(i8, b, undefined, odd16);
    const s = ae * be + ao * bo;
    return @intCast(@min(@max(s, @as(@Vector(16, i32), @splat(-32768))), @as(@Vector(16, i32), @splat(32767))));
}

/// i16 x i16 -> pairwise-summed i32 (vpmaddwd).
pub inline fn maddwd(a: V16i16, b: V16i16) V8i32 {
    if (has_avx2) return @"llvm.x86.avx2.pmadd.wd"(a, b);
    const ae: V8i32 = @shuffle(i16, a, undefined, even8);
    const ao: V8i32 = @shuffle(i16, a, undefined, odd8);
    const be: V8i32 = @shuffle(i16, b, undefined, even8);
    const bo: V8i32 = @shuffle(i16, b, undefined, odd8);
    return ae * be +% ao * bo;
}

pub inline fn toF(v: V8i32) V8f {
    return @floatFromInt(v);
}

const has_f16c = builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .f16c);

/// fp16 -> fp32. Plain @floatCast lowers to vpinsrw+vcvtph2ps and LLVM merges
/// into whatever register it likes (often the loop accumulator), creating a
/// false loop-carried dependency; vmovd fully overwrites the register.
pub inline fn f16f(h: f16) f32 {
    if (has_f16c) {
        return asm ("vmovd %[b], %[r]\n\tvcvtph2ps %[r], %[r]"
            : [r] "=&x" (-> f32),
            : [b] "r" (@as(u32, @as(u16, @bitCast(h)))),
        );
    }
    return @floatCast(h);
}

const ones16: V16i16 = @splat(1);

/// Decode the 12-byte packed 6-bit scales/mins of Q4_K/Q5_K: [0..8] scales, [8..16] mins.
pub inline fn scalesK4(s: *const [12]u8) [16]u8 {
    const kmask1: u32 = 0x3f3f3f3f;
    const kmask2: u32 = 0x0f0f0f0f;
    const kmask3: u32 = 0x03030303;
    var u: [4]u32 = .{
        std.mem.readInt(u32, s[0..4], .little),
        std.mem.readInt(u32, s[4..8], .little),
        std.mem.readInt(u32, s[8..12], .little),
        0,
    };
    u[3] = ((u[2] >> 4) & kmask2) | (((u[1] >> 6) & kmask3) << 4);
    const uaux = u[1] & kmask1;
    u[1] = (u[2] & kmask2) | (((u[0] >> 6) & kmask3) << 4);
    u[2] = uaux;
    u[0] &= kmask1;
    return @bitCast(u);
}

// ---------------------------------------------------------------------------
// Activation quantization

pub fn quantizeQ8K(x: []const f32, out: []Q8K) void {
    std.debug.assert(x.len == out.len * 256);
    for (out, 0..) |*b, i| {
        const xs = x[i * 256 ..][0..256];
        var amax: V8f = @splat(0);
        inline for (0..32) |j| amax = @max(amax, @abs(@as(V8f, xs[j * 8 ..][0..8].*)));
        const m = @reduce(.Max, amax);
        if (m == 0) {
            b.* = std.mem.zeroes(Q8K);
            continue;
        }
        const id: V8f = @splat(127.0 / m);
        b.d = m / 127.0;
        inline for (0..16) |j| {
            const v0: V8f = xs[j * 16 ..][0..8].*;
            const v1: V8f = xs[j * 16 + 8 ..][0..8].*;
            const q0: V8i32 = @intFromFloat(@round(v0 * id));
            const q1: V8i32 = @intFromFloat(@round(v1 * id));
            b.qs[j * 16 ..][0..8].* = @as(@Vector(8, i8), @intCast(q0));
            b.qs[j * 16 + 8 ..][0..8].* = @as(@Vector(8, i8), @intCast(q1));
            b.bsums[j] = @intCast(@reduce(.Add, q0 + q1));
        }
        inline for (0..8) |j| b.bsums32[j] = b.bsums[2 * j] + b.bsums[2 * j + 1];
    }
}

pub fn quantizeQ8B(x: []const f32, out: []Q8B) void {
    std.debug.assert(x.len == out.len * 32);
    for (out, 0..) |*b, i| {
        const xs = x[i * 32 ..][0..32];
        var amax: V8f = @splat(0);
        inline for (0..4) |j| amax = @max(amax, @abs(@as(V8f, xs[j * 8 ..][0..8].*)));
        const m = @reduce(.Max, amax);
        if (m == 0) {
            b.* = std.mem.zeroes(Q8B);
            continue;
        }
        const id: V8f = @splat(127.0 / m);
        b.d = m / 127.0;
        var sum: V8i32 = @splat(0);
        inline for (0..4) |j| {
            const q: V8i32 = @intFromFloat(@round(@as(V8f, xs[j * 8 ..][0..8].*) * id));
            sum += q;
            b.qs[j * 8 ..][0..8].* = @as(@Vector(8, i8), @intCast(q));
        }
        b.s = b.d * @as(f32, @floatFromInt(@reduce(.Add, sum)));
    }
}

// ---------------------------------------------------------------------------
// Dot products: one weight row (nb blocks) against NT activation rows.
// NT == 1 is the generation path. For NT > 1 every weight block is unpacked
// once into small stack arrays and reused by all tokens, which keeps register
// pressure flat and lets the loads fold into vpmaddubsw/vpmaddwd.

const dup16: [16]i32 = .{ 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7 };

/// 256 unpacked unsigned weights plus one i16 scale vector per 32 values.
const Unpacked = struct {
    q: [8]V32u8 align(32),
    s: [8]V16i16 align(32),

    /// Integer dot of the unpacked block with one activation block.
    inline fn dot(u: *const Unpacked, y: *const Q8K) V8i32 {
        var a: V8i32 = maddwd(maddubs(u.q[0], y.qs[0..32].*), u.s[0]);
        inline for (1..8) |j| a += maddwd(maddubs(u.q[j], y.qs[j * 32 ..][0..32].*), u.s[j]);
        return a;
    }
};

inline fn unpackK45(comptime is5: bool, x: anytype, sm: *const [16]u8, u: *Unpacked) void {
    const m4: V32u8 = @splat(0x0f);
    const one: V32u8 = @splat(1);
    inline for (0..4) |j| {
        const q: V32u8 = x.qs[j * 32 ..][0..32].*;
        var lo = q & m4;
        var hi = q >> @splat(4);
        if (is5) {
            const hb: V32u8 = x.qh;
            lo |= ((hb >> @splat(2 * j)) & one) << @splat(4);
            hi |= ((hb >> @splat(2 * j + 1)) & one) << @splat(4);
        }
        u.q[2 * j] = lo;
        u.q[2 * j + 1] = hi;
        u.s[2 * j] = @splat(@as(i16, sm[2 * j]));
        u.s[2 * j + 1] = @splat(@as(i16, sm[2 * j + 1]));
    }
}

fn dotK45(comptime is5: bool, comptime NT: usize, row: [*]const u8, acts: [NT][*]const Q8K, nb: usize) [NT]f32 {
    const B = if (is5) BlockQ5_K else BlockQ4_K;
    const w: [*]const B = @ptrCast(@alignCast(row));
    var accf: [NT]V8f = @splat(@splat(0));
    var u: Unpacked = undefined;
    for (0..nb) |i| {
        const x = &w[i];
        const sm = scalesK4(&x.scales);
        const mv: @Vector(8, i16) = @intCast(@as(@Vector(8, u8), sm[8..16].*));
        const mins: V16i16 = @shuffle(i16, mv, undefined, dup16);
        unpackK45(is5, x, &sm, &u);
        const d = f16f(x.d);
        const dmin = f16f(x.dmin);
        // Runtime loop on purpose: keeps the unpacked block in memory instead
        // of NT-way unrolling that spills registers.
        var t: usize = 0;
        while (t < NT) : (t += 1) {
            const y = &acts[t][i];
            const a = u.dot(y);
            accf[t] = @mulAdd(V8f, @splat(d * y.d), toF(a), accf[t]);
            accf[t] = @mulAdd(V8f, @splat(-dmin * y.d), toF(maddwd(y.bsums, mins)), accf[t]);
        }
    }
    var r: [NT]f32 = undefined;
    inline for (0..NT) |t| r[t] = @reduce(.Add, accf[t]);
    return r;
}

/// Lanes 0..7 = v[2k], lanes 8..15 = v[2k+1].
inline fn scalePair(v: V16i16, comptime k: usize) V16i16 {
    return @shuffle(i16, v, undefined, [16]i32{ 2 * k, 2 * k, 2 * k, 2 * k, 2 * k, 2 * k, 2 * k, 2 * k, 2 * k + 1, 2 * k + 1, 2 * k + 1, 2 * k + 1, 2 * k + 1, 2 * k + 1, 2 * k + 1, 2 * k + 1 });
}

fn dotQ6K(comptime NT: usize, row: [*]const u8, acts: [NT][*]const Q8K, nb: usize) [NT]f32 {
    const w: [*]const BlockQ6_K = @ptrCast(@alignCast(row));
    const m4: V32u8 = @splat(0x0f);
    const m3: V32u8 = @splat(3);
    var accf: [NT]V8f = @splat(@splat(0));
    var u: Unpacked = undefined;
    for (0..nb) |i| {
        const x = &w[i];
        const scv: V16i16 = @intCast(@as(@Vector(16, i8), x.scales));
        inline for (0..2) |half| {
            const h: V32u8 = x.qh[half * 32 ..][0..32].*;
            const la: V32u8 = x.ql[half * 64 ..][0..32].*;
            const lb: V32u8 = x.ql[half * 64 + 32 ..][0..32].*;
            const o = half * 4;
            u.q[o + 0] = (la & m4) | ((h & m3) << @splat(4));
            u.q[o + 1] = (lb & m4) | (((h >> @splat(2)) & m3) << @splat(4));
            u.q[o + 2] = (la >> @splat(4)) | (((h >> @splat(4)) & m3) << @splat(4));
            u.q[o + 3] = (lb >> @splat(4)) | ((h >> @splat(6)) << @splat(4));
            inline for (0..4) |k| u.s[o + k] = scalePair(scv, o + k);
        }
        const d = f16f(x.d);
        var t: usize = 0;
        while (t < NT) : (t += 1) {
            const y = &acts[t][i];
            // Values are stored +32; remove the offset via the per-16 sums.
            const a = u.dot(y) - (maddwd(scv, y.bsums) << @splat(5));
            accf[t] = @mulAdd(V8f, @splat(d * y.d), toF(a), accf[t]);
        }
    }
    var r: [NT]f32 = undefined;
    inline for (0..NT) |t| r[t] = @reduce(.Add, accf[t]);
    return r;
}

const concat16: [32]i32 = blk: {
    var m: [32]i32 = undefined;
    for (0..16) |k| {
        m[k] = k;
        m[16 + k] = ~@as(i32, k);
    }
    break :blk m;
};

fn dotQ4_0(comptime NT: usize, row: [*]const u8, acts: [NT][*]const Q8B, nb: usize) [NT]f32 {
    const w: [*]const BlockQ4_0 = @ptrCast(@alignCast(row));
    var accf: [NT]V8f = @splat(@splat(0));
    var off: [NT]f32 = @splat(0);
    for (0..nb) |i| {
        const x = &w[i];
        const qs: @Vector(16, u8) = x.qs;
        const lo = qs & @as(@Vector(16, u8), @splat(0x0f));
        const hi = qs >> @splat(4);
        const q: V32u8 = @shuffle(u8, lo, hi, concat16);
        const d = f16f(x.d);
        inline for (0..NT) |t| {
            const y = &acts[t][i];
            accf[t] += @as(V8f, @splat(d * y.d)) * toF(maddwd(maddubs(q, y.qs), ones16));
            off[t] += d * y.s;
        }
    }
    var r: [NT]f32 = undefined;
    inline for (0..NT) |t| r[t] = @reduce(.Add, accf[t]) - 8.0 * off[t];
    return r;
}

fn dotQ8_0(comptime NT: usize, row: [*]const u8, acts: [NT][*]const Q8B, nb: usize) [NT]f32 {
    const w: [*]const BlockQ8_0 = @ptrCast(@alignCast(row));
    var accf: [NT]V8f = @splat(@splat(0));
    for (0..nb) |i| {
        const x = &w[i];
        const xq: V32i8 = x.qs;
        const neg = xq < @as(V32i8, @splat(0));
        const ax: V32u8 = @abs(xq);
        const d = f16f(x.d);
        inline for (0..NT) |t| {
            const y = &acts[t][i];
            const yq: V32i8 = y.qs;
            const sy = @select(i8, neg, @as(V32i8, @splat(0)) -% yq, yq);
            accf[t] += @as(V8f, @splat(d * y.d)) * toF(maddwd(maddubs(ax, sy), ones16));
        }
    }
    var r: [NT]f32 = undefined;
    inline for (0..NT) |t| r[t] = @reduce(.Add, accf[t]);
    return r;
}

fn dotF32(w: [*]const f32, a: [*]const f32, n: usize) f32 {
    var acc: V16f = @splat(0);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) acc += @as(V16f, w[i..][0..16].*) * @as(V16f, a[i..][0..16].*);
    var s = @reduce(.Add, acc);
    while (i < n) : (i += 1) s += w[i] * a[i];
    return s;
}

pub fn dotF16(w: [*]const f16, a: [*]const f32, n: usize) f32 {
    var acc: V16f = @splat(0);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        const h: @Vector(16, f16) = w[i..][0..16].*;
        acc += @as(V16f, @floatCast(h)) * @as(V16f, a[i..][0..16].*);
    }
    var s = @reduce(.Add, acc);
    while (i < n) : (i += 1) s += f16f(w[i]) * a[i];
    return s;
}

fn dotBF16(w: [*]const u16, a: [*]const f32, n: usize) f32 {
    var acc: V16f = @splat(0);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        const h: @Vector(16, u32) = @as(@Vector(16, u16), w[i..][0..16].*);
        acc += @as(V16f, @bitCast(h << @splat(16))) * @as(V16f, a[i..][0..16].*);
    }
    var s = @reduce(.Add, acc);
    while (i < n) : (i += 1) s += bf16f(w[i]) * a[i];
    return s;
}

inline fn bf16f(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

inline fn castActs(comptime T: type, comptime NT: usize, acts: [NT][*]const u8) [NT][*]const T {
    var r: [NT][*]const T = undefined;
    inline for (0..NT) |k| r[k] = @ptrCast(@alignCast(acts[k]));
    return r;
}

/// Dot products of one weight row of type `t` (n elements) with NT activation
/// rows already converted to `actKind(t)`.
pub inline fn dotN(comptime t: GgmlType, comptime NT: usize, row: [*]const u8, acts: [NT][*]const u8, n: usize) [NT]f32 {
    switch (t) {
        .q4_k => return dotK45(false, NT, row, castActs(Q8K, NT, acts), n / 256),
        .q5_k => return dotK45(true, NT, row, castActs(Q8K, NT, acts), n / 256),
        .q6_k => return dotQ6K(NT, row, castActs(Q8K, NT, acts), n / 256),
        .q4_0 => return dotQ4_0(NT, row, castActs(Q8B, NT, acts), n / 32),
        .q8_0 => return dotQ8_0(NT, row, castActs(Q8B, NT, acts), n / 32),
        .f32, .f16, .bf16 => {
            var r: [NT]f32 = undefined;
            inline for (0..NT) |k| r[k] = switch (t) {
                .f32 => dotF32(@ptrCast(@alignCast(row)), @ptrCast(@alignCast(acts[k])), n),
                .f16 => dotF16(@ptrCast(@alignCast(row)), @ptrCast(@alignCast(acts[k])), n),
                .bf16 => dotBF16(@ptrCast(@alignCast(row)), @ptrCast(@alignCast(acts[k])), n),
                else => unreachable,
            };
            return r;
        },
        else => unreachable,
    }
}

pub inline fn dot(comptime t: GgmlType, row: [*]const u8, act: [*]const u8, n: usize) f32 {
    return dotN(t, 1, row, .{act}, n)[0];
}

// ---------------------------------------------------------------------------
// Dequantization (embedding lookup, tests, non-matmul tensors).

pub fn dequantRow(t: GgmlType, src: [*]const u8, dst: []f32) void {
    const n = dst.len;
    switch (t) {
        .f32 => @memcpy(dst, @as([*]const f32, @ptrCast(@alignCast(src)))[0..n]),
        .f16 => for (dst, @as([*]const f16, @ptrCast(@alignCast(src)))[0..n]) |*d, s| {
            d.* = f16f(s);
        },
        .bf16 => for (dst, @as([*]const u16, @ptrCast(@alignCast(src)))[0..n]) |*d, s| {
            d.* = bf16f(s);
        },
        .q4_0 => {
            const bs: [*]const BlockQ4_0 = @ptrCast(@alignCast(src));
            for (0..n / 32) |i| {
                const d = f16f(bs[i].d);
                for (0..16) |l| {
                    dst[i * 32 + l] = d * (@as(f32, @floatFromInt(bs[i].qs[l] & 0xf)) - 8);
                    dst[i * 32 + 16 + l] = d * (@as(f32, @floatFromInt(bs[i].qs[l] >> 4)) - 8);
                }
            }
        },
        .q8_0 => {
            const bs: [*]const BlockQ8_0 = @ptrCast(@alignCast(src));
            for (0..n / 32) |i| {
                const d = f16f(bs[i].d);
                for (0..32) |l| dst[i * 32 + l] = d * @as(f32, @floatFromInt(bs[i].qs[l]));
            }
        },
        .q4_k, .q5_k => {
            const is5 = t == .q5_k;
            const bsz: usize = if (is5) 176 else 144;
            for (0..n / 256) |i| {
                const b = src + i * bsz;
                const d = f16f(@bitCast(std.mem.readInt(u16, b[0..2], .little)));
                const dmin = f16f(@bitCast(std.mem.readInt(u16, b[2..4], .little)));
                const sm = scalesK4(b[4..16]);
                const qh: []const u8 = if (is5) b[16..48] else &.{};
                const qs = if (is5) b[48..176] else b[16..144];
                const y = dst[i * 256 ..][0..256];
                for (0..4) |j| {
                    const d1 = d * @as(f32, @floatFromInt(sm[2 * j]));
                    const m1 = dmin * @as(f32, @floatFromInt(sm[8 + 2 * j]));
                    const d2 = d * @as(f32, @floatFromInt(sm[2 * j + 1]));
                    const m2 = dmin * @as(f32, @floatFromInt(sm[8 + 2 * j + 1]));
                    for (0..32) |l| {
                        var lo: u8 = qs[j * 32 + l] & 0xf;
                        var hi: u8 = qs[j * 32 + l] >> 4;
                        if (is5) {
                            lo |= ((qh[l] >> @intCast(2 * j)) & 1) << 4;
                            hi |= ((qh[l] >> @intCast(2 * j + 1)) & 1) << 4;
                        }
                        y[j * 64 + l] = d1 * @as(f32, @floatFromInt(lo)) - m1;
                        y[j * 64 + 32 + l] = d2 * @as(f32, @floatFromInt(hi)) - m2;
                    }
                }
            }
        },
        .q6_k => {
            const bs: [*]const BlockQ6_K = @ptrCast(@alignCast(src));
            for (0..n / 256) |i| {
                const x = &bs[i];
                const d = f16f(x.d);
                const y = dst[i * 256 ..][0..256];
                for (0..2) |half| {
                    const ql = x.ql[half * 64 ..];
                    const qh = x.qh[half * 32 ..];
                    const sc = x.scales[half * 8 ..];
                    const yy = y[half * 128 ..];
                    for (0..32) |l| {
                        const is = l / 16;
                        const q1: i32 = @as(i32, (ql[l] & 0xf) | ((qh[l] & 3) << 4)) - 32;
                        const q2: i32 = @as(i32, (ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const q3: i32 = @as(i32, (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const q4: i32 = @as(i32, (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        yy[l] = d * @as(f32, @floatFromInt(sc[is] * q1));
                        yy[l + 32] = d * @as(f32, @floatFromInt(sc[is + 2] * q2));
                        yy[l + 64] = d * @as(f32, @floatFromInt(sc[is + 4] * q3));
                        yy[l + 96] = d * @as(f32, @floatFromInt(sc[is + 6] * q4));
                    }
                }
            }
        },
        else => unreachable,
    }
}

// ---------------------------------------------------------------------------
// Tests: every quantized kernel must match dequant(w) . dequant(act).

fn dequantAct(kind: ActKind, act: []const u8, dst: []f32) void {
    switch (kind) {
        .q8k => {
            const bs: [*]const Q8K = @ptrCast(@alignCast(act.ptr));
            for (dst, 0..) |*d, i| d.* = bs[i / 256].d * @as(f32, @floatFromInt(bs[i / 256].qs[i % 256]));
        },
        .q8b => {
            const bs: [*]const Q8B = @ptrCast(@alignCast(act.ptr));
            for (dst, 0..) |*d, i| d.* = bs[i / 32].d * @as(f32, @floatFromInt(bs[i / 32].qs[i % 32]));
        },
        .f32 => @memcpy(dst, @as([*]const f32, @ptrCast(@alignCast(act.ptr)))[0..dst.len]),
    }
}

fn checkKernel(comptime t: GgmlType) !void {
    const n = 1024;
    const l = t.layout().?;
    var prng = std.Random.DefaultPrng.init(@intFromEnum(t) * 7919 + 1);
    const r = prng.random();

    var wbuf: [n * 4]u8 align(64) = undefined;
    const wbytes = wbuf[0 .. n / l.block * l.bytes];
    r.bytes(wbytes);
    // Overwrite fp16 scale fields with sane values so results stay finite.
    const nb = n / l.block;
    for (0..nb) |b| {
        const blk = wbytes[b * l.bytes ..][0..l.bytes];
        const sane = struct {
            fn put(dst: []u8, v: f32) void {
                std.mem.writeInt(u16, dst[0..2], @bitCast(@as(f16, @floatCast(v))), .little);
            }
        }.put;
        switch (t) {
            .q4_0, .q8_0 => sane(blk[0..2], 0.01 + r.float(f32) * 0.05),
            .q4_k, .q5_k => {
                sane(blk[0..2], 0.01 + r.float(f32) * 0.05);
                sane(blk[2..4], 0.01 + r.float(f32) * 0.05);
            },
            .q6_k => sane(blk[208..210], 0.01 + r.float(f32) * 0.05),
            .f16 => for (0..l.block) |k| sane(blk[k * 2 ..][0..2], r.floatNorm(f32)),
            .f32 => std.mem.writeInt(u32, blk[0..4], @bitCast(r.floatNorm(f32)), .little),
            .bf16 => std.mem.writeInt(u16, blk[0..2], @truncate(@as(u32, @bitCast(r.floatNorm(f32))) >> 16), .little),
            else => unreachable,
        }
    }

    // NT activation rows, each checked against its own f64 reference.
    const NT = 3;
    var abuf: [NT][n / 256 * @sizeOf(Q8K) + n * 4]u8 align(64) = undefined;
    var ptrs: [NT][*]const u8 = undefined;
    const kind = actKind(t);
    var wf: [n]f32 = undefined;
    dequantRow(t, wbytes.ptr, &wf);
    var refs: [NT]f64 = undefined;
    var mags: [NT]f64 = undefined;
    for (0..NT) |k| {
        var x: [n]f32 = undefined;
        for (&x) |*v| v.* = r.floatNorm(f32);
        switch (kind) {
            .q8k => quantizeQ8K(&x, @as([*]Q8K, @ptrCast(@alignCast(&abuf[k])))[0 .. n / 256]),
            .q8b => quantizeQ8B(&x, @as([*]Q8B, @ptrCast(@alignCast(&abuf[k])))[0 .. n / 32]),
            .f32 => @memcpy(@as([*]f32, @ptrCast(@alignCast(&abuf[k])))[0..n], &x),
        }
        ptrs[k] = &abuf[k];
        var af: [n]f32 = undefined;
        dequantAct(kind, &abuf[k], &af);
        refs[k] = 0;
        mags[k] = 0;
        for (wf, af) |a, b| {
            refs[k] += @as(f64, a) * b;
            mags[k] += @abs(@as(f64, a) * b);
        }
    }
    const multi = dotN(t, NT, wbytes.ptr, ptrs, n);
    for (0..NT) |k| {
        const single = dot(t, wbytes.ptr, ptrs[k], n);
        for ([_]f32{ single, multi[k] }) |got| {
            const err = @abs(@as(f64, got) - refs[k]);
            if (err > 1e-4 * mags[k] + 1e-5) {
                std.debug.print("{s}[{d}]: got {d} ref {d} (mag {d})\n", .{ @tagName(t), k, got, refs[k], mags[k] });
                return error.KernelMismatch;
            }
        }
    }
}

test "quantized kernels match dequantized reference" {
    inline for (.{ GgmlType.q4_0, .q8_0, .q4_k, .q5_k, .q6_k, .f16, .bf16, .f32 }) |t| try checkKernel(t);
}

test "Q8K quantization round-trips within half a step" {
    var x: [512]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(3);
    for (&x) |*v| v.* = prng.random().floatNorm(f32) * 3;
    var q: [2]Q8K = undefined;
    quantizeQ8K(&x, &q);
    for (x, 0..) |v, i| {
        const b = q[i / 256];
        const back = b.d * @as(f32, @floatFromInt(b.qs[i % 256]));
        try std.testing.expect(@abs(back - v) <= b.d * 0.5001);
    }
    var s: i32 = 0;
    for (q[1].qs[16..32]) |v| s += v;
    try std.testing.expectEqual(s, q[1].bsums[1]);
}
