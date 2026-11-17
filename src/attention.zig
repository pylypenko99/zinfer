// attention.zig - FlashAttention, FlashDecoding, KV Cache Packing & Quantization (f16, q8_0, q4_0, q2_0, q2_k)
// Zero-dependency pure Zig module.

const std = @import("std");
const quant = @import("quant.zig");

pub const CacheType = enum(u32) {
    f16 = 0,
    q8_0 = 1,
    q4_0 = 2,
    q2_0 = 3,
    q2_k = 4,
};

pub const kv_tile: usize = 16;
pub const max_q: usize = 8;
pub const max_head_dim: usize = 512;
pub const max_spec: usize = 8;
pub const max_flash_chunks: usize = 16;

pub const PartialStats = struct {
    m: [max_q]f32,
    l: [max_q]f32,
};

const V8f = @Vector(8, f32);
const V16f = @Vector(16, f32);
const V8i32 = @Vector(8, i32);

/// Fast Walsh-Hadamard Transform (FWHT) for power-of-2 length vectors.
/// Rotates activations by orthogonal matrix H_d (H_d * H_d^T = I),
/// preserving all inner products and norms exactly (<H x, H y> == <x, y>)
/// while diffusing heavy-tail outlier dimensions into Gaussian statistics via CLT.
pub inline fn fwhtSimd(x: []f32) void {
    const n = x.len;
    var len: usize = 1;
    while (len < n) : (len *= 2) {
        if (len >= 8) {
            var i: usize = 0;
            while (i < n) : (i += 2 * len) {
                var j: usize = 0;
                while (j < len) : (j += 8) {
                    const u: V8f = x[i + j ..][0..8].*;
                    const v: V8f = x[i + len + j ..][0..8].*;
                    x[i + j ..][0..8].* = u + v;
                    x[i + len + j ..][0..8].* = u - v;
                }
            }
        } else {
            var i: usize = 0;
            while (i < n) : (i += 2 * len) {
                for (0..len) |j| {
                    const u = x[i + j];
                    const v = x[i + len + j];
                    x[i + j] = u + v;
                    x[i + len + j] = u - v;
                }
            }
        }
    }
    const norm_v: V8f = @splat(1.0 / @sqrt(@as(f32, @floatFromInt(n))));
    var j: usize = 0;
    while (j < n) : (j += 8) {
        x[j..][0..8].* = @as(V8f, x[j..][0..8].*) * norm_v;
    }
}

/// Cephes-style expf on 8 lanes (rel. error ~1e-7).
inline fn expV(x_in: V8f) V8f {
    var x = @min(@max(x_in, @as(V8f, @splat(-87.3))), @as(V8f, @splat(88.3)));
    const fx = @floor(@mulAdd(V8f, x, @as(V8f, @splat(1.44269504088896341)), @as(V8f, @splat(0.5))));
    x = @mulAdd(V8f, -fx, @as(V8f, @splat(0.693359375)), x);
    x = @mulAdd(V8f, -fx, @as(V8f, @splat(-2.12194440e-4)), x);
    var y: V8f = @splat(1.9875691500e-4);
    y = @mulAdd(V8f, y, x, @as(V8f, @splat(1.3981999507e-3)));
    y = @mulAdd(V8f, y, x, @as(V8f, @splat(8.3334519073e-3)));
    y = @mulAdd(V8f, y, x, @as(V8f, @splat(4.1665795894e-2)));
    y = @mulAdd(V8f, y, x, @as(V8f, @splat(1.6666665459e-1)));
    y = @mulAdd(V8f, y, x, @as(V8f, @splat(5.0000001201e-1)));
    const x2 = x * x;
    y = @mulAdd(V8f, y, x2, x + @as(V8f, @splat(1.0)));
    const n: @Vector(8, i32) = @intFromFloat(fx);
    const p2: V8f = @bitCast((n + @as(@Vector(8, i32), @splat(127))) << @splat(23));
    return y * p2;
}

/// PMM (Polynomial Maximization Method) scale estimation for 4-bit block quantization.
/// Evaluates sample moments (m1..m4) and Kunchenko's non-Gaussian variance reduction
/// factor g2 = 1 - gamma3^2 / (2 + gamma4) with asymmetric shift \hat{\theta} = m1 + h2 * (m2c - c2)
/// where h2 = -c3 / (c4 + 2*c2^2), adapting the clipping boundary to skewed activation distributions.
pub inline fn pmmScaleBlock32_q4(xs: *const [32]f32, amax: f32) f32 {
    if (amax <= 1e-12) return 0;

    var s1: V8f = @splat(0);
    var s2: V8f = @splat(0);
    var s3: V8f = @splat(0);
    var s4: V8f = @splat(0);
    inline for (0..4) |j| {
        const v = @as(V8f, xs[j * 8 ..][0..8].*);
        s1 += v;
        const v2 = v * v;
        s2 += v2;
        s3 += v2 * v;
        s4 += v2 * v2;
    }
    const m1 = @reduce(.Add, s1) / 32.0;
    const m2 = @reduce(.Add, s2) / 32.0;
    const var_c = @max(m2 - m1 * m1, 1e-12);
    const sigma = @sqrt(var_c);
    const m3 = @reduce(.Add, s3) / 32.0;
    const m4 = @reduce(.Add, s4) / 32.0;

    const c3 = m3 - 3.0 * m1 * var_c - m1 * m1 * m1;
    const c4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 * m1 * m1 * m1 - 3.0 * var_c * var_c;

    const gamma3 = c3 / (sigma * var_c);
    const gamma4 = c4 / (var_c * var_c);
    const denom = 2.0 + gamma4;
    const g2 = if (denom > 1e-4) std.math.clamp(1.0 - (gamma3 * gamma3) / denom, 0.1, 1.0) else 1.0;

    const h2_denom = c4 + 2.0 * var_c * var_c;
    const h2 = if (h2_denom > 1e-12) -c3 / h2_denom else 0.0;
    const theta_shift = m1 + h2 * var_c;
    const eta = @abs(theta_shift) / sigma;
    const asym_penalty = std.math.clamp(0.035 * eta, 0.0, 0.05);

    const factor: f32 = std.math.clamp(0.935 + 0.035 * g2 - asym_penalty, 0.88, 0.98);
    return amax * factor;
}

/// PMM scale estimation for 2-bit block quantization (q2_0 / q2_k).
/// 2-bit quantization has only 4 levels (-3, -1, 1, 3); adapting the scale
/// to sample variance, kurtosis, and asymmetric shift prevents single outliers
/// from compressing all inner representations.
pub inline fn pmmScaleBlock32_q2(xs: *const [32]f32, amax: f32) f32 {
    if (amax <= 1e-12) return 0;

    var s1: V8f = @splat(0);
    var s2: V8f = @splat(0);
    var s3: V8f = @splat(0);
    var s4: V8f = @splat(0);
    inline for (0..4) |j| {
        const v = @as(V8f, xs[j * 8 ..][0..8].*);
        s1 += v;
        const v2 = v * v;
        s2 += v2;
        s3 += v2 * v;
        s4 += v2 * v2;
    }
    const m1 = @reduce(.Add, s1) / 32.0;
    const m2 = @reduce(.Add, s2) / 32.0;
    const var_c = @max(m2 - m1 * m1, 1e-12);
    const sigma = @sqrt(var_c);
    const m3 = @reduce(.Add, s3) / 32.0;
    const m4 = @reduce(.Add, s4) / 32.0;

    const c3 = m3 - 3.0 * m1 * var_c - m1 * m1 * m1;
    const c4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 * m1 * m1 * m1 - 3.0 * var_c * var_c;

    const gamma3 = c3 / (sigma * var_c);
    const gamma4 = c4 / (var_c * var_c);
    const denom = 2.0 + gamma4;
    const g2 = if (denom > 1e-4) std.math.clamp(1.0 - (gamma3 * gamma3) / denom, 0.1, 1.0) else 1.0;

    const h2_denom = c4 + 2.0 * var_c * var_c;
    const h2 = if (h2_denom > 1e-12) -c3 / h2_denom else 0.0;
    const theta_shift = m1 + h2 * var_c;
    const eta = @abs(theta_shift) / sigma;
    const asym_penalty = std.math.clamp(0.05 * eta, 0.0, 0.07);

    const factor: f32 = std.math.clamp(0.830 + 0.050 * g2 - asym_penalty, 0.75, 0.92);
    return amax * factor;
}

/// Quantize one 32-value block: max-abs i8 values written with stride
/// `stride` starting at `vals`, f16 scale at `sc`.
pub inline fn quantBlock32(xs: *const [32]f32, vals: [*]u8, stride: usize, sc: *align(1) f16) void {
    var amax: V8f = @splat(0);
    inline for (0..4) |j| amax = @max(amax, @abs(@as(V8f, xs[j * 8 ..][0..8].*)));
    const m_raw = @reduce(.Max, amax);
    if (m_raw == 0) {
        inline for (0..32) |i| vals[i * stride] = 0;
        sc.* = 0;
        return;
    }
    const m = m_raw * 0.996;
    const id: V8f = @splat(127.0 / m);
    sc.* = @floatCast(m / 127.0);
    inline for (0..4) |j| {
        const v = @as(V8f, xs[j * 8 ..][0..8].*) * id;
        const q: V8i32 = @intFromFloat(@round(v));
        const qc = @min(@max(q, @as(V8i32, @splat(-127))), @as(V8i32, @splat(127)));
        const qb: @Vector(8, i8) = @intCast(qc);
        inline for (0..8) |i| vals[(j * 8 + i) * stride] = @bitCast(qb[i]);
    }
}

/// Quantize one 32-value block to symmetric 4-bit nibbles:
/// values in [-7, 7] represented as 4-bit two's complement in [0, 15],
/// written to `q4[0..32]`, with f16 scale written to `sc`.
pub inline fn quantBlock32_q4(xs: *const [32]f32, q4: *[32]u8, sc: *align(1) f16) void {
    var amax: V8f = @splat(0);
    inline for (0..4) |j| amax = @max(amax, @abs(@as(V8f, xs[j * 8 ..][0..8].*)));
    const m_raw = @reduce(.Max, amax);
    if (m_raw == 0) {
        @memset(q4, 0);
        sc.* = 0;
        return;
    }
    const m = pmmScaleBlock32_q4(xs, m_raw);
    const id: V8f = @splat(7.0 / m);
    sc.* = @floatCast(m / 7.0);
    inline for (0..4) |j| {
        const v = @as(V8f, xs[j * 8 ..][0..8].*) * id;
        const q: V8i32 = @intFromFloat(@round(v));
        const qc = @min(@max(q, @as(V8i32, @splat(-7))), @as(V8i32, @splat(7)));
        const qb: @Vector(8, i8) = @intCast(qc);
        const qu: [8]u8 = @as(@Vector(8, u8), @bitCast(qb)) & @as(@Vector(8, u8), @splat(0x0F));
        q4[j * 8 ..][0..8].* = qu;
    }
}

/// Quantize one 32-value block to symmetric 2-bit levels:
/// values in {-3, -1, 1, 3} mapped to codes {0, 1, 2, 3},
/// written to `q2[0..32]`, with f16 scale written to `sc`.
pub inline fn quantBlock32_q2(xs: *const [32]f32, q2: *[32]u8, sc: *align(1) f16) void {
    var amax: V8f = @splat(0);
    inline for (0..4) |j| amax = @max(amax, @abs(@as(V8f, xs[j * 8 ..][0..8].*)));
    const m_raw = @reduce(.Max, amax);
    if (m_raw == 0) {
        @memset(q2, 1);
        sc.* = 0;
        return;
    }
    const m = pmmScaleBlock32_q2(xs, m_raw);
    const s = m / 3.0;
    sc.* = @floatCast(s);
    const id: V8f = @splat(1.0 / s);
    inline for (0..4) |j| {
        const v = @as(V8f, xs[j * 8 ..][0..8].*) * id;
        const mapped = @round(v * @as(V8f, @splat(0.5)) + @as(V8f, @splat(1.5)));
        const qi: V8i32 = @intFromFloat(mapped);
        const qc = @min(@max(qi, @as(V8i32, @splat(0))), @as(V8i32, @splat(3)));
        const qb: @Vector(8, u8) = @intCast(qc);
        q2[j * 8 ..][0..8].* = qb;
    }
}

/// Online-softmax attention of NQ queries against one KV head; query i sees
/// positions [lo[i], np[i]). Writes the normalized outputs to `out`.
/// CACHE selects f16, q8_0, or q4_0 cache layout; kt/vr are raw bytes.
/// If window > 0, the KV cache is addressed as a ring buffer of w_tiles tiles.
pub fn attendBlock(
    comptime NQ: usize,
    comptime CACHE: CacheType,
    hd: usize,
    kt: [*]const u8,
    vr: [*]const u8,
    q: [NQ][*]const f32,
    lo: [NQ]usize,
    np: [NQ]usize,
    out: [NQ][*]f32,
    scale: f32,
    window: usize,
    w_tiles: usize,
) void {
    attendBlockChunk(NQ, CACHE, hd, kt, vr, q, lo, np, out, scale, window, w_tiles, 0, std.math.maxInt(usize), null);
}

pub fn attendBlockChunk(
    comptime NQ: usize,
    comptime CACHE: CacheType,
    hd: usize,
    kt: [*]const u8,
    vr: [*]const u8,
    q: [NQ][*]const f32,
    lo: [NQ]usize,
    np: [NQ]usize,
    out: [NQ][*]f32,
    scale: f32,
    window: usize,
    w_tiles: usize,
    tile_start: usize,
    tile_end: usize,
    partial_stats: ?*PartialStats,
) void {
    const k_tile_bytes: usize = switch (CACHE) {
        .f16 => hd * 32,
        .q8_0 => hd * 17,
        .q4_0 => hd * 9,
        .q2_0, .q2_k => hd * 5,
    };
    const v_tile_bytes: usize = switch (CACHE) {
        .f16 => hd * 32,
        .q8_0 => hd * 17,
        .q4_0, .q2_k => hd * 9,
        .q2_0 => hd * 5,
    };
    const nblocks = hd / 32;
    var o: [NQ][max_head_dim]f32 align(32) = undefined;
    var m: [NQ]f32 = @splat(-std.math.inf(f32));
    var l: [NQ]f32 = @splat(0);
    inline for (0..NQ) |i| @memset(o[i][0..hd], 0);
    var maxnp: usize = 0;
    var minlo: usize = std.math.maxInt(usize);
    inline for (0..NQ) |i| {
        maxnp = @max(maxnp, np[i]);
        minlo = @min(minlo, lo[i]);
    }
    const ntiles = (maxnp + kv_tile - 1) / kv_tile;

    const j_start = @max(minlo / kv_tile, tile_start);
    const j_end = @min(ntiles, tile_end);

    for (j_start..j_end) |j| {
        const tile_idx = if (window > 0) j % w_tiles else j;
        const tile = kt + tile_idx * k_tile_bytes;
        if (j + 1 < j_end) {
            const next_idx = if (window > 0) (j + 1) % w_tiles else (j + 1);
            const nk = kt + next_idx * k_tile_bytes;
            const nv = vr + next_idx * v_tile_bytes;
            inline for (0..8) |cl| {
                @prefetch(nk + cl * 64, .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(nv + cl * 64, .{ .rw = .read, .locality = 3, .cache = .data });
            }
        }
        var s: [NQ]V16f = @splat(@splat(0));
        switch (CACHE) {
            .q4_0 => {
                comptime var qs: usize = 0;
                inline while (qs < NQ) : (qs += 4) {
                    const q_sub = comptime @min(4, NQ - qs);
                    var bp = tile;
                    var b: usize = 0;
                    var s0_acc: [q_sub]V8f = @splat(@splat(0));
                    var s1_acc: [q_sub]V8f = @splat(@splat(0));
                    while (b < nblocks) : (b += 1) {
                        if (b + 1 < nblocks) @prefetch(bp + 288, .{ .rw = .read, .locality = 3, .cache = .data });
                        const ksc: [*]const f16 = @ptrCast(@alignCast(bp + 256));
                        const sv: V16f = @floatCast(@as(@Vector(16, f16), ksc[0..kv_tile].*));
                        const sv0: V8f = @as([16]f32, sv)[0..8].*;
                        const sv1: V8f = @as([16]f32, sv)[8..16].*;
                        var b_s0: [q_sub]V8f = @splat(@splat(0));
                        var b_s1: [q_sub]V8f = @splat(@splat(0));
                        inline for (0..32) |dp| {
                            const d = b * 32 + dp;
                            const raw8: [8]u8 = bp[dp * 8 ..][0..8].*;
                            const u8_8: @Vector(8, u8) = raw8;
                            const lo_u = u8_8 & @as(@Vector(8, u8), @splat(0x0F));
                            const hi_u = (u8_8 >> @as(@Vector(8, u8), @splat(4))) & @as(@Vector(8, u8), @splat(0x0F));
                            const s0: @Vector(8, i8) = @as(@Vector(8, i8), @bitCast(lo_u ^ @as(@Vector(8, u8), @splat(0x08)))) -% @as(@Vector(8, i8), @splat(8));
                            const s1: @Vector(8, i8) = @as(@Vector(8, i8), @bitCast(hi_u ^ @as(@Vector(8, u8), @splat(0x08)))) -% @as(@Vector(8, i8), @splat(8));
                            const kf0: V8f = @floatFromInt(s0);
                            const kf1: V8f = @floatFromInt(s1);
                            inline for (0..q_sub) |qi_idx| {
                                const i = qs + qi_idx;
                                const qi: V8f = @splat(q[i][d]);
                                b_s0[qi_idx] = @mulAdd(V8f, qi, kf0, b_s0[qi_idx]);
                                b_s1[qi_idx] = @mulAdd(V8f, qi, kf1, b_s1[qi_idx]);
                            }
                        }
                        inline for (0..q_sub) |qi_idx| {
                            s0_acc[qi_idx] = @mulAdd(V8f, b_s0[qi_idx], sv0, s0_acc[qi_idx]);
                            s1_acc[qi_idx] = @mulAdd(V8f, b_s1[qi_idx], sv1, s1_acc[qi_idx]);
                        }
                        bp += 288;
                    }
                    inline for (0..q_sub) |qi_idx| {
                        const i = qs + qi_idx;
                        var s_arr: [16]f32 = undefined;
                        s_arr[0..8].* = s0_acc[qi_idx];
                        s_arr[8..16].* = s1_acc[qi_idx];
                        s[i] = s_arr;
                    }
                }
            },
            .q2_0, .q2_k => {
                comptime var qs: usize = 0;
                inline while (qs < NQ) : (qs += 4) {
                    const q_sub = comptime @min(4, NQ - qs);
                    var bp = tile;
                    var b: usize = 0;
                    var s0_acc: [q_sub]V8f = @splat(@splat(0));
                    var s1_acc: [q_sub]V8f = @splat(@splat(0));
                    while (b < nblocks) : (b += 1) {
                        if (b + 1 < nblocks) @prefetch(bp + 160, .{ .rw = .read, .locality = 3, .cache = .data });
                        const ksc: [*]const f16 = @ptrCast(@alignCast(bp + 128));
                        const sv: V16f = @floatCast(@as(@Vector(16, f16), ksc[0..kv_tile].*));
                        const sv0: V8f = @as([16]f32, sv)[0..8].*;
                        const sv1: V8f = @as([16]f32, sv)[8..16].*;
                        var b_s0: [q_sub]V8f = @splat(@splat(0));
                        var b_s1: [q_sub]V8f = @splat(@splat(0));
                        inline for (0..32) |dp| {
                            const d = b * 32 + dp;
                            const raw4: [4]u8 = bp[dp * 4 ..][0..4].*;
                            const u8_4: @Vector(4, u8) = raw4;
                            const b0 = u8_4 & @as(@Vector(4, u8), @splat(0x03));
                            const b1 = (u8_4 >> @as(@Vector(4, u8), @splat(2))) & @as(@Vector(4, u8), @splat(0x03));
                            const b2 = (u8_4 >> @as(@Vector(4, u8), @splat(4))) & @as(@Vector(4, u8), @splat(0x03));
                            const b3 = (u8_4 >> @as(@Vector(4, u8), @splat(6))) & @as(@Vector(4, u8), @splat(0x03));

                            const s0_u: @Vector(8, u8) = @shuffle(u8, b0, b1, [8]i32{ 0, 1, 2, 3, -1, -2, -3, -4 });
                            const s1_u: @Vector(8, u8) = @shuffle(u8, b2, b3, [8]i32{ 0, 1, 2, 3, -1, -2, -3, -4 });

                            const kf0: V8f = @as(V8f, @floatFromInt(s0_u)) * @as(V8f, @splat(2.0)) - @as(V8f, @splat(3.0));
                            const kf1: V8f = @as(V8f, @floatFromInt(s1_u)) * @as(V8f, @splat(2.0)) - @as(V8f, @splat(3.0));

                            inline for (0..q_sub) |qi_idx| {
                                const i = qs + qi_idx;
                                const qi: V8f = @splat(q[i][d]);
                                b_s0[qi_idx] = @mulAdd(V8f, qi, kf0, b_s0[qi_idx]);
                                b_s1[qi_idx] = @mulAdd(V8f, qi, kf1, b_s1[qi_idx]);
                            }
                        }
                        inline for (0..q_sub) |qi_idx| {
                            s0_acc[qi_idx] = @mulAdd(V8f, b_s0[qi_idx], sv0, s0_acc[qi_idx]);
                            s1_acc[qi_idx] = @mulAdd(V8f, b_s1[qi_idx], sv1, s1_acc[qi_idx]);
                        }
                        bp += 160;
                    }
                    inline for (0..q_sub) |qi_idx| {
                        const i = qs + qi_idx;
                        var s_arr: [16]f32 = undefined;
                        s_arr[0..8].* = s0_acc[qi_idx];
                        s_arr[8..16].* = s1_acc[qi_idx];
                        s[i] = s_arr;
                    }
                }
            },
            .q8_0 => {
                comptime var qs: usize = 0;
                inline while (qs < NQ) : (qs += 4) {
                    const q_sub = comptime @min(4, NQ - qs);
                    var bp = tile;
                    var b: usize = 0;
                    var s_sub: [q_sub]V16f = @splat(@splat(0));
                    while (b < nblocks) : (b += 1) {
                        const ksc: [*]const f16 = @ptrCast(@alignCast(bp + 512));
                        const sv: V16f = @floatCast(@as(@Vector(16, f16), ksc[0..kv_tile].*));
                        var b_acc: [q_sub]V16f = @splat(@splat(0));
                        inline for (0..32) |dp| {
                            const d = b * 32 + dp;
                            const kf: V16f = @floatFromInt(@as(@Vector(16, i8), @bitCast(@as(@Vector(16, u8), bp[dp * 16 ..][0..16].*))));
                            inline for (0..q_sub) |qi_idx| {
                                const i = qs + qi_idx;
                                b_acc[qi_idx] = @mulAdd(V16f, @splat(q[i][d]), kf, b_acc[qi_idx]);
                            }
                        }
                        inline for (0..q_sub) |qi_idx| {
                            s_sub[qi_idx] = @mulAdd(V16f, b_acc[qi_idx], sv, s_sub[qi_idx]);
                        }
                        bp += 544;
                    }
                    inline for (0..q_sub) |qi_idx| {
                        s[qs + qi_idx] = s_sub[qi_idx];
                    }
                }
            },
            .f16 => {
                comptime var qs: usize = 0;
                inline while (qs < NQ) : (qs += 4) {
                    const q_sub = comptime @min(4, NQ - qs);
                    const kf16: [*]const f16 = @ptrCast(@alignCast(tile));
                    var s_sub: [q_sub]V16f = @splat(@splat(0));
                    for (0..hd) |d| {
                        const kf: V16f = @floatCast(@as(@Vector(16, f16), kf16[d * kv_tile ..][0..kv_tile].*));
                        inline for (0..q_sub) |qi_idx| {
                            const i = qs + qi_idx;
                            s_sub[qi_idx] = @mulAdd(V16f, @splat(q[i][d]), kf, s_sub[qi_idx]);
                        }
                    }
                    inline for (0..q_sub) |qi_idx| {
                        s[qs + qi_idx] = s_sub[qi_idx];
                    }
                }
            },
        }
        // Mask, then fold the tile into the running softmax of each query.
        var p: [NQ][kv_tile]f32 = undefined;
        const base = j * kv_tile;
        var has_active_p = false;
        inline for (0..NQ) |i| {
            if (base >= np[i] or base + kv_tile <= lo[i]) {
                p[i] = @splat(0);
            } else {
                var sv = s[i] * @as(V16f, @splat(scale));
                if (base < lo[i] or base + kv_tile > np[i]) {
                    const lane = std.simd.iota(u32, kv_tile) + @as(@Vector(kv_tile, u32), @splat(@intCast(base)));
                    const visible = (lane < @as(@Vector(kv_tile, u32), @splat(@intCast(np[i])))) & (lane >= @as(@Vector(kv_tile, u32), @splat(@intCast(lo[i]))));
                    sv = @select(f32, visible, sv, @as(V16f, @splat(-std.math.inf(f32))));
                }
                const tmax = @reduce(.Max, sv);
                if (tmax == -std.math.inf(f32) or (m[i] != -std.math.inf(f32) and m[i] - tmax > 13.0)) {
                    p[i] = @splat(0);
                } else {
                    has_active_p = true;
                    const m_new = @max(m[i], tmax);
                    const corr = @exp(m[i] - m_new);
                    const mv: V8f = @splat(m_new);
                    const e0 = expV(@as(V8f, @as([kv_tile]f32, sv)[0..8].*) - mv);
                    const e1 = expV(@as(V8f, @as([kv_tile]f32, sv)[8..16].*) - mv);
                    p[i][0..8].* = e0;
                    p[i][8..16].* = e1;
                    l[i] = l[i] * corr + @reduce(.Add, e0 + e1);
                    if (corr != 1) {
                        const cv: V8f = @splat(corr);
                        var d: usize = 0;
                        while (d + 16 <= hd) : (d += 16) {
                            o[i][d..][0..8].* = @as(V8f, o[i][d..][0..8].*) * cv;
                            o[i][d + 8 ..][0..8].* = @as(V8f, o[i][d + 8 ..][0..8].*) * cv;
                        }
                        while (d < hd) : (d += 8) o[i][d..][0..8].* = @as(V8f, o[i][d..][0..8].*) * cv;
                    }
                    m[i] = m_new;
                }
            }
        }

        if (!has_active_p) continue;

        // o += p . V for this tile, 8 or 16 output dims at a time kept in registers.
        const nvalid = @min(kv_tile, maxnp - base);
        switch (CACHE) {
            .q4_0, .q2_k => {
                const vt = vr + tile_idx * v_tile_bytes;
                comptime var qs: usize = 0;
                inline while (qs < NQ) : (qs += 4) {
                    const q_sub = comptime @min(4, NQ - qs);
                    var ps: [q_sub][kv_tile]f32 = undefined;
                    var b: usize = 0;
                    while (b < nblocks) : (b += 1) {
                        if (b + 1 < nblocks) @prefetch(vt + 8 * hd + (b + 1) * 32, .{ .rw = .read, .locality = 3, .cache = .data });
                        const vsc: [*]const f16 = @ptrCast(@alignCast(vt + 8 * hd + b * 32));
                        const sv: V16f = @floatCast(@as(@Vector(16, f16), vsc[0..kv_tile].*));
                        inline for (0..q_sub) |qi_idx| {
                            const i = qs + qi_idx;
                            ps[qi_idx][0..kv_tile].* = @as(V16f, p[i][0..kv_tile].*) * sv;
                        }
                        inline for (0..2) |c2| {
                            const d = b * 32 + c2 * 16;
                            var acc0: [q_sub]V8f = undefined;
                            var acc1: [q_sub]V8f = undefined;
                            inline for (0..q_sub) |qi_idx| {
                                const i = qs + qi_idx;
                                acc0[qi_idx] = o[i][d..][0..8].*;
                                acc1[qi_idx] = o[i][d + 8 ..][0..8].*;
                            }
                            for (0..nvalid) |n| {
                                const raw8: [8]u8 = vt[n * (hd / 2) + b * 16 + c2 * 8 ..][0..8].*;
                                const u8_8: @Vector(8, u8) = raw8;
                                const lo_u = u8_8 & @as(@Vector(8, u8), @splat(0x0F));
                                const hi_u = (u8_8 >> @as(@Vector(8, u8), @splat(4))) & @as(@Vector(8, u8), @splat(0x0F));
                                const s0: @Vector(8, i8) = @as(@Vector(8, i8), @bitCast(lo_u ^ @as(@Vector(8, u8), @splat(0x08)))) -% @as(@Vector(8, i8), @splat(8));
                                const s1: @Vector(8, i8) = @as(@Vector(8, i8), @bitCast(hi_u ^ @as(@Vector(8, u8), @splat(0x08)))) -% @as(@Vector(8, i8), @splat(8));
                                const vv0: V8f = @floatFromInt(s0);
                                const vv1: V8f = @floatFromInt(s1);
                                inline for (0..q_sub) |qi_idx| {
                                    const psi: V8f = @splat(ps[qi_idx][n]);
                                    acc0[qi_idx] = @mulAdd(V8f, psi, vv0, acc0[qi_idx]);
                                    acc1[qi_idx] = @mulAdd(V8f, psi, vv1, acc1[qi_idx]);
                                }
                            }
                            inline for (0..q_sub) |qi_idx| {
                                const i = qs + qi_idx;
                                o[i][d..][0..8].* = acc0[qi_idx];
                                o[i][d + 8 ..][0..8].* = acc1[qi_idx];
                            }
                        }
                    }
                }
            },
            .q2_0 => {
                const vt = vr + tile_idx * v_tile_bytes;
                comptime var qs: usize = 0;
                inline while (qs < NQ) : (qs += 4) {
                    const q_sub = comptime @min(4, NQ - qs);
                    var ps: [q_sub][kv_tile]f32 = undefined;
                    var b: usize = 0;
                    while (b < nblocks) : (b += 1) {
                        if (b + 1 < nblocks) @prefetch(vt + 4 * hd + (b + 1) * 32, .{ .rw = .read, .locality = 3, .cache = .data });
                        const vsc: [*]const f16 = @ptrCast(@alignCast(vt + 4 * hd + b * 32));
                        const sv: V16f = @floatCast(@as(@Vector(16, f16), vsc[0..kv_tile].*));
                        inline for (0..q_sub) |qi_idx| {
                            const i = qs + qi_idx;
                            ps[qi_idx][0..kv_tile].* = @as(V16f, p[i][0..kv_tile].*) * sv;
                        }
                        const d_base = b * 32;
                        var acc0: [q_sub]V8f = undefined;
                        var acc1: [q_sub]V8f = undefined;
                        var acc2: [q_sub]V8f = undefined;
                        var acc3: [q_sub]V8f = undefined;
                        inline for (0..q_sub) |qi_idx| {
                            const i = qs + qi_idx;
                            acc0[qi_idx] = o[i][d_base..][0..8].*;
                            acc1[qi_idx] = o[i][d_base + 8 ..][0..8].*;
                            acc2[qi_idx] = o[i][d_base + 16 ..][0..8].*;
                            acc3[qi_idx] = o[i][d_base + 24 ..][0..8].*;
                        }
                        for (0..nvalid) |n| {
                            const raw8: [8]u8 = vt[n * (hd / 4) + b * 8 ..][0..8].*;
                            const u8_8: @Vector(8, u8) = raw8;
                            const d0_7 = u8_8 & @as(@Vector(8, u8), @splat(0x03));
                            const d8_15 = (u8_8 >> @as(@Vector(8, u8), @splat(2))) & @as(@Vector(8, u8), @splat(0x03));
                            const d16_23 = (u8_8 >> @as(@Vector(8, u8), @splat(4))) & @as(@Vector(8, u8), @splat(0x03));
                            const d24_31 = (u8_8 >> @as(@Vector(8, u8), @splat(6))) & @as(@Vector(8, u8), @splat(0x03));

                            const vv0: V8f = @as(V8f, @floatFromInt(d0_7)) * @as(V8f, @splat(2.0)) - @as(V8f, @splat(3.0));
                            const vv1: V8f = @as(V8f, @floatFromInt(d8_15)) * @as(V8f, @splat(2.0)) - @as(V8f, @splat(3.0));
                            const vv2: V8f = @as(V8f, @floatFromInt(d16_23)) * @as(V8f, @splat(2.0)) - @as(V8f, @splat(3.0));
                            const vv3: V8f = @as(V8f, @floatFromInt(d24_31)) * @as(V8f, @splat(2.0)) - @as(V8f, @splat(3.0));

                            inline for (0..q_sub) |qi_idx| {
                                const psi: V8f = @splat(ps[qi_idx][n]);
                                acc0[qi_idx] = @mulAdd(V8f, psi, vv0, acc0[qi_idx]);
                                acc1[qi_idx] = @mulAdd(V8f, psi, vv1, acc1[qi_idx]);
                                acc2[qi_idx] = @mulAdd(V8f, psi, vv2, acc2[qi_idx]);
                                acc3[qi_idx] = @mulAdd(V8f, psi, vv3, acc3[qi_idx]);
                            }
                        }
                        inline for (0..q_sub) |qi_idx| {
                            const i = qs + qi_idx;
                            o[i][d_base..][0..8].* = acc0[qi_idx];
                            o[i][d_base + 8 ..][0..8].* = acc1[qi_idx];
                            o[i][d_base + 16 ..][0..8].* = acc2[qi_idx];
                            o[i][d_base + 24 ..][0..8].* = acc3[qi_idx];
                        }
                    }
                }
            },
            .q8_0 => {
                const vt = vr + tile_idx * v_tile_bytes;
                comptime var qs: usize = 0;
                inline while (qs < NQ) : (qs += 4) {
                    const q_sub = comptime @min(4, NQ - qs);
                    var ps: [q_sub][kv_tile]f32 = undefined;
                    var b: usize = 0;
                    while (b < nblocks) : (b += 1) {
                        const vsc: [*]const f16 = @ptrCast(@alignCast(vt + 16 * hd + b * 32));
                        const sv: V16f = @floatCast(@as(@Vector(16, f16), vsc[0..kv_tile].*));
                        inline for (0..q_sub) |qi_idx| {
                            const i = qs + qi_idx;
                            ps[qi_idx][0..kv_tile].* = @as(V16f, p[i][0..kv_tile].*) * sv;
                        }
                        inline for (0..2) |c2| {
                            const d = b * 32 + c2 * 16;
                            var acc0: [q_sub]V8f = undefined;
                            var acc1: [q_sub]V8f = undefined;
                            inline for (0..q_sub) |qi_idx| {
                                const i = qs + qi_idx;
                                acc0[qi_idx] = o[i][d..][0..8].*;
                                acc1[qi_idx] = o[i][d + 8 ..][0..8].*;
                            }
                            for (0..nvalid) |n| {
                                const v16: [16]u8 = vt[n * hd + d ..][0..16].*;
                                const vv0: V8f = @floatFromInt(@as(@Vector(8, i8), @bitCast(@as(@Vector(8, u8), v16[0..8].*))));
                                const vv1: V8f = @floatFromInt(@as(@Vector(8, i8), @bitCast(@as(@Vector(8, u8), v16[8..16].*))));
                                inline for (0..q_sub) |qi_idx| {
                                    const psi: V8f = @splat(ps[qi_idx][n]);
                                    acc0[qi_idx] = @mulAdd(V8f, psi, vv0, acc0[qi_idx]);
                                    acc1[qi_idx] = @mulAdd(V8f, psi, vv1, acc1[qi_idx]);
                                }
                            }
                            inline for (0..q_sub) |qi_idx| {
                                const i = qs + qi_idx;
                                o[i][d..][0..8].* = acc0[qi_idx];
                                o[i][d + 8 ..][0..8].* = acc1[qi_idx];
                            }
                        }
                    }
                }
            },
            .f16 => {
                const vt: [*]const f16 = @ptrCast(@alignCast(vr + tile_idx * v_tile_bytes));
                comptime var qs: usize = 0;
                inline while (qs < NQ) : (qs += 4) {
                    const q_sub = comptime @min(4, NQ - qs);
                    var d: usize = 0;
                    while (d + 16 <= hd) : (d += 16) {
                        var acc0: [q_sub]V8f = undefined;
                        var acc1: [q_sub]V8f = undefined;
                        inline for (0..q_sub) |qi_idx| {
                            const i = qs + qi_idx;
                            acc0[qi_idx] = o[i][d..][0..8].*;
                            acc1[qi_idx] = o[i][d + 8 ..][0..8].*;
                        }
                        for (0..nvalid) |n| {
                            const v16: [16]f16 = vt[n * hd + d ..][0..16].*;
                            const vv0: V8f = @floatCast(@as(@Vector(8, f16), v16[0..8].*));
                            const vv1: V8f = @floatCast(@as(@Vector(8, f16), v16[8..16].*));
                            inline for (0..q_sub) |qi_idx| {
                                const i = qs + qi_idx;
                                const pi: V8f = @splat(p[i][n]);
                                acc0[qi_idx] = @mulAdd(V8f, pi, vv0, acc0[qi_idx]);
                                acc1[qi_idx] = @mulAdd(V8f, pi, vv1, acc1[qi_idx]);
                            }
                        }
                        inline for (0..q_sub) |qi_idx| {
                            const i = qs + qi_idx;
                            o[i][d..][0..8].* = acc0[qi_idx];
                            o[i][d + 8 ..][0..8].* = acc1[qi_idx];
                        }
                    }
                    while (d < hd) : (d += 8) {
                        var acc: [q_sub]V8f = undefined;
                        inline for (0..q_sub) |qi_idx| {
                            const i = qs + qi_idx;
                            acc[qi_idx] = o[i][d..][0..8].*;
                        }
                        for (0..nvalid) |n| {
                            const vv: V8f = @floatCast(@as(@Vector(8, f16), vt[n * hd + d ..][0..8].*));
                            inline for (0..q_sub) |qi_idx| {
                                const i = qs + qi_idx;
                                acc[qi_idx] = @mulAdd(V8f, @splat(p[i][n]), vv, acc[qi_idx]);
                            }
                        }
                        inline for (0..q_sub) |qi_idx| {
                            const i = qs + qi_idx;
                            o[i][d..][0..8].* = acc[qi_idx];
                        }
                    }
                }
            },
        }
    }
    if (partial_stats) |ps| {
        inline for (0..NQ) |i| {
            ps.m[i] = m[i];
            ps.l[i] = l[i];
            var d: usize = 0;
            while (d + 16 <= hd) : (d += 16) {
                out[i][d..][0..8].* = o[i][d..][0..8].*;
                out[i][d + 8 ..][0..8].* = o[i][d + 8 ..][0..8].*;
            }
            while (d < hd) : (d += 8) out[i][d..][0..8].* = o[i][d..][0..8].*;
        }
    } else {
        inline for (0..NQ) |i| {
            const inv: V8f = @splat(if (l[i] > 0) 1.0 / l[i] else 0);
            var d: usize = 0;
            while (d + 16 <= hd) : (d += 16) {
                out[i][d..][0..8].* = @as(V8f, o[i][d..][0..8].*) * inv;
                out[i][d + 8 ..][0..8].* = @as(V8f, o[i][d + 8 ..][0..8].*) * inv;
            }
            while (d < hd) : (d += 8) out[i][d..][0..8].* = @as(V8f, o[i][d..][0..8].*) * inv;
        }
    }
}

pub fn reduceFlashDecoding(
    out_ptr: [*]f32,
    flash_o: [*]const f32,
    flash_m: []const f32,
    flash_l: []const f32,
    n_heads: usize,
    head_dim: usize,
    q_dim: usize,
    n_chunks: usize,
) void {
    const hd = head_dim;
    for (0..n_heads) |h| {
        var max_m: f32 = -std.math.inf(f32);
        for (0..n_chunks) |ci| {
            max_m = @max(max_m, flash_m[ci * n_heads + h]);
        }

        if (max_m == -std.math.inf(f32)) {
            @memset(out_ptr[h * hd ..][0..hd], 0);
            continue;
        }

        var total_l: f32 = 0;
        var alpha: [max_flash_chunks]f32 = undefined;
        for (0..n_chunks) |ci| {
            const mc = flash_m[ci * n_heads + h];
            if (mc > -std.math.inf(f32)) {
                const a = @exp(mc - max_m);
                alpha[ci] = a;
                total_l += flash_l[ci * n_heads + h] * a;
            } else {
                alpha[ci] = 0;
            }
        }

        const inv_l: V8f = @splat(if (total_l > 0) 1.0 / total_l else 0.0);
        const cur_out = out_ptr + h * hd;

        var d: usize = 0;
        while (d + 16 <= hd) : (d += 16) {
            var acc0: V8f = @splat(0);
            var acc1: V8f = @splat(0);
            for (0..n_chunks) |ci| {
                const a = alpha[ci];
                if (a == 0) continue;
                const av: V8f = @splat(a);
                const chunk_ptr = flash_o + ci * q_dim + h * hd + d;
                acc0 = @mulAdd(V8f, av, chunk_ptr[0..8].*, acc0);
                acc1 = @mulAdd(V8f, av, (chunk_ptr + 8)[0..8].*, acc1);
            }
            cur_out[d..][0..8].* = acc0 * inv_l;
            (cur_out + d + 8)[0..8].* = acc1 * inv_l;
        }
        while (d < hd) : (d += 8) {
            var acc: V8f = @splat(0);
            for (0..n_chunks) |ci| {
                const a = alpha[ci];
                if (a == 0) continue;
                const av: V8f = @splat(a);
                const chunk_ptr = flash_o + ci * q_dim + h * hd + d;
                acc = @mulAdd(V8f, av, chunk_ptr[0..8].*, acc);
            }
            cur_out[d..][0..8].* = acc * inv_l;
        }
    }
}

// --------------------------------------------------------------------------
// Tests

test "tiled online-softmax attention matches naive reference" {
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    inline for (.{ 64, 128 }) |hd| {
        const npos = 53;
        const ntiles = (npos + kv_tile - 1) / kv_tile;
        var kt: [ntiles * hd * kv_tile]f16 = undefined;
        var vr: [ntiles * kv_tile * hd]f16 = undefined;
        var kf: [npos][hd]f32 = undefined;
        var vf: [npos][hd]f32 = undefined;
        for (&kt) |*x| x.* = std.math.nan(f16);
        for (0..npos) |p| for (0..hd) |d| {
            const kk: f16 = @floatCast(r.floatNorm(f32));
            const vv: f16 = @floatCast(r.floatNorm(f32));
            kt[(p / kv_tile) * hd * kv_tile + d * kv_tile + p % kv_tile] = kk;
            vr[p * hd + d] = vv;
            kf[p][d] = kk;
            vf[p][d] = vv;
        };
        var q: [3][hd]f32 = undefined;
        for (&q) |*qq| for (qq) |*x| {
            x.* = r.floatNorm(f32) * 2;
        };
        const lo = [3]usize{ 21, 0, 0 };
        const np = [3]usize{ npos, 17, 1 };
        var out: [3][hd]f32 = undefined;
        const scale: f32 = 0.125;
        attendBlock(3, .f16, hd, @ptrCast(&kt), @ptrCast(&vr), .{ &q[0], &q[1], &q[2] }, lo, np, .{ &out[0], &out[1], &out[2] }, scale, 0, 0);
        for (0..3) |i| {
            var s: [npos]f64 = undefined;
            var mx: f64 = -std.math.inf(f64);
            for (lo[i]..np[i]) |p| {
                var dot: f64 = 0;
                for (0..hd) |d| dot += @as(f64, q[i][d]) * kf[p][d];
                s[p] = dot * scale;
                mx = @max(mx, s[p]);
            }
            var sum: f64 = 0;
            for (lo[i]..np[i]) |p| {
                s[p] = @exp(s[p] - mx);
                sum += s[p];
            }
            for (0..hd) |d| {
                var want: f64 = 0;
                for (lo[i]..np[i]) |p| want += s[p] / sum * vf[p][d];
                try std.testing.expectApproxEqAbs(want, out[i][d], 1e-4);
            }
        }
    }
}

test "quantized tiled online-softmax attention matches dequantized reference" {
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    inline for (.{ 64, 128 }) |hd| {
        const npos = 53;
        const ntiles = (npos + kv_tile - 1) / kv_tile;
        const nblocks = hd / 32;
        const tile_bytes = hd * 17;
        var kt: [ntiles * tile_bytes]u8 align(32) = undefined;
        var vr: [ntiles * tile_bytes]u8 align(32) = undefined;
        @memset(&kt, 0xFF);
        @memset(&vr, 0xFF);
        var ksrc: [npos][hd]f32 = undefined;
        var vsrc: [npos][hd]f32 = undefined;
        for (&ksrc) |*row| for (row) |*x| {
            x.* = r.floatNorm(f32);
        };
        for (&vsrc) |*row| for (row) |*x| {
            x.* = r.floatNorm(f32);
        };
        for (0..npos) |p| {
            const tile = p / kv_tile;
            const pl = p % kv_tile;
            for (0..nblocks) |b| {
                quantBlock32(ksrc[p][b * 32 ..][0..32], kt[tile * tile_bytes + b * 544 + pl ..].ptr, kv_tile, @ptrCast(&kt[tile * tile_bytes + b * 544 + 512 + pl * 2]));
                quantBlock32(vsrc[p][b * 32 ..][0..32], vr[tile * tile_bytes + pl * hd + b * 32 ..].ptr, 1, @ptrCast(&vr[tile * tile_bytes + 16 * hd + b * 32 + pl * 2]));
            }
        }
        var kf: [npos][hd]f32 = undefined;
        var vf: [npos][hd]f32 = undefined;
        for (0..npos) |p| {
            const tile = p / kv_tile;
            const pl = p % kv_tile;
            for (0..nblocks) |b| {
                const ks: f32 = @as(*align(1) f16, @ptrCast(&kt[tile * tile_bytes + b * 544 + 512 + pl * 2])).*;
                const vs: f32 = @as(*align(1) f16, @ptrCast(&vr[tile * tile_bytes + 16 * hd + b * 32 + pl * 2])).*;
                for (0..32) |dp| {
                    kf[p][b * 32 + dp] = @as(f32, @floatFromInt(@as(i8, @bitCast(kt[tile * tile_bytes + b * 544 + dp * 16 + pl])))) * ks;
                    vf[p][b * 32 + dp] = @as(f32, @floatFromInt(@as(i8, @bitCast(vr[tile * tile_bytes + pl * hd + b * 32 + dp])))) * vs;
                }
            }
        }
        var q: [3][hd]f32 = undefined;
        for (&q) |*qq| for (qq) |*x| {
            x.* = r.floatNorm(f32) * 2;
        };
        const lo = [3]usize{ 21, 0, 0 };
        const np = [3]usize{ npos, 17, 1 };
        var out: [3][hd]f32 = undefined;
        const scale: f32 = 0.125;
        attendBlock(3, .q8_0, hd, &kt, &vr, .{ &q[0], &q[1], &q[2] }, lo, np, .{ &out[0], &out[1], &out[2] }, scale, 0, 0);
        for (0..3) |i| {
            var s: [npos]f64 = undefined;
            var mx: f64 = -std.math.inf(f64);
            for (lo[i]..np[i]) |p| {
                var dot: f64 = 0;
                for (0..hd) |d| dot += @as(f64, q[i][d]) * kf[p][d];
                s[p] = dot * scale;
                mx = @max(mx, s[p]);
            }
            var sum: f64 = 0;
            for (lo[i]..np[i]) |p| {
                s[p] = @exp(s[p] - mx);
                sum += s[p];
            }
            for (0..hd) |d| {
                var want: f64 = 0;
                for (lo[i]..np[i]) |p| want += s[p] / sum * vf[p][d];
                try std.testing.expectApproxEqAbs(want, out[i][d], 1e-4);
            }
        }
    }
}

test "q4_0 quantized tiled online-softmax attention matches dequantized reference" {
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    inline for (.{ 64, 128 }) |hd| {
        const npos = 53;
        const ntiles = (npos + kv_tile - 1) / kv_tile;
        const nblocks = hd / 32;
        const tile_bytes = hd * 9;
        var kt: [ntiles * tile_bytes]u8 align(32) = undefined;
        var vr: [ntiles * tile_bytes]u8 align(32) = undefined;
        @memset(&kt, 0xFF);
        @memset(&vr, 0xFF);
        var ksrc: [npos][hd]f32 = undefined;
        var vsrc: [npos][hd]f32 = undefined;
        for (&ksrc) |*row| for (row) |*x| {
            x.* = r.floatNorm(f32);
        };
        for (&vsrc) |*row| for (row) |*x| {
            x.* = r.floatNorm(f32);
        };
        for (0..npos) |p| {
            const tile = p / kv_tile;
            const pl = p % kv_tile;
            for (0..nblocks) |b| {
                var q4_k: [32]u8 = undefined;
                var sc_k: f16 = undefined;
                quantBlock32_q4(ksrc[p][b * 32 ..][0..32], &q4_k, &sc_k);

                const k_bp = kt[tile * tile_bytes + b * 288 ..].ptr;
                if (pl == 0) @memset(k_bp[0..256], 0);
                @as(*align(1) f16, @ptrCast(k_bp + 256 + pl * 2)).* = sc_k;
                if (pl < 8) {
                    inline for (0..32) |dp| {
                        k_bp[dp * 8 + pl] = (k_bp[dp * 8 + pl] & 0xF0) | q4_k[dp];
                    }
                } else {
                    inline for (0..32) |dp| {
                        k_bp[dp * 8 + (pl - 8)] = (k_bp[dp * 8 + (pl - 8)] & 0x0F) | (q4_k[dp] << 4);
                    }
                }

                var q4_v: [32]u8 = undefined;
                var sc_v: f16 = undefined;
                quantBlock32_q4(vsrc[p][b * 32 ..][0..32], &q4_v, &sc_v);

                const vt = vr[tile * tile_bytes ..].ptr;
                @as(*align(1) f16, @ptrCast(vt + 8 * hd + b * 32 + pl * 2)).* = sc_v;
                const v_dst = vt + pl * (hd / 2) + b * 16;
                const q_lo: @Vector(8, u8) = q4_v[0..8].*;
                const q_hi: @Vector(8, u8) = q4_v[8..16].*;
                const v0: [8]u8 = q_lo | (q_hi << @as(@Vector(8, u8), @splat(4)));
                const q_lo2: @Vector(8, u8) = q4_v[16..24].*;
                const q_hi2: @Vector(8, u8) = q4_v[24..32].*;
                const v1: [8]u8 = q_lo2 | (q_hi2 << @as(@Vector(8, u8), @splat(4)));
                @memcpy(v_dst[0..8], &v0);
                @memcpy(v_dst[8..16], &v1);
            }
        }
        var kf: [npos][hd]f32 = undefined;
        var vf: [npos][hd]f32 = undefined;
        for (0..npos) |p| {
            const tile = p / kv_tile;
            const pl = p % kv_tile;
            for (0..nblocks) |b| {
                const k_bp = kt[tile * tile_bytes + b * 288 ..].ptr;
                const ks: f32 = @as(*align(1) f16, @ptrCast(k_bp + 256 + pl * 2)).*;
                for (0..32) |dp| {
                    const u: u8 = if (pl < 8)
                        (k_bp[dp * 8 + pl] & 0x0F)
                    else
                        ((k_bp[dp * 8 + (pl - 8)] >> 4) & 0x0F);
                    const s: i8 = @as(i8, @bitCast(u ^ 8)) -% 8;
                    kf[p][b * 32 + dp] = @as(f32, @floatFromInt(s)) * ks;
                }

                const vt = vr[tile * tile_bytes ..].ptr;
                const vs: f32 = @as(*align(1) f16, @ptrCast(vt + 8 * hd + b * 32 + pl * 2)).*;
                const row = vt + pl * (hd / 2) + b * 16;
                for (0..8) |j| {
                    const v_lo = row[j] & 0x0F;
                    const v_hi = (row[j] >> 4) & 0x0F;
                    const s0: i8 = @as(i8, @bitCast(v_lo ^ 8)) -% 8;
                    const s1: i8 = @as(i8, @bitCast(v_hi ^ 8)) -% 8;
                    vf[p][b * 32 + j] = @as(f32, @floatFromInt(s0)) * vs;
                    vf[p][b * 32 + 8 + j] = @as(f32, @floatFromInt(s1)) * vs;
                }
                for (0..8) |j| {
                    const v_lo2 = row[8 + j] & 0x0F;
                    const v_hi2 = (row[8 + j] >> 4) & 0x0F;
                    const s2: i8 = @as(i8, @bitCast(v_lo2 ^ 8)) -% 8;
                    const s3: i8 = @as(i8, @bitCast(v_hi2 ^ 8)) -% 8;
                    vf[p][b * 32 + 16 + j] = @as(f32, @floatFromInt(s2)) * vs;
                    vf[p][b * 32 + 24 + j] = @as(f32, @floatFromInt(s3)) * vs;
                }
            }
        }
        var q: [3][hd]f32 = undefined;
        for (&q) |*qq| for (qq) |*x| {
            x.* = r.floatNorm(f32) * 2;
        };
        const lo = [3]usize{ 21, 0, 0 };
        const np = [3]usize{ npos, 17, 1 };
        var out: [3][hd]f32 = undefined;
        const scale: f32 = 0.125;
        attendBlock(3, .q4_0, hd, &kt, &vr, .{ &q[0], &q[1], &q[2] }, lo, np, .{ &out[0], &out[1], &out[2] }, scale, 0, 0);
        for (0..3) |i| {
            var s: [npos]f64 = undefined;
            var mx: f64 = -std.math.inf(f64);
            for (lo[i]..np[i]) |p| {
                var dot: f64 = 0;
                for (0..hd) |d| dot += @as(f64, q[i][d]) * kf[p][d];
                s[p] = dot * scale;
                mx = @max(mx, s[p]);
            }
            var sum: f64 = 0;
            for (lo[i]..np[i]) |p| {
                s[p] = @exp(s[p] - mx);
                sum += s[p];
            }
            for (0..hd) |d| {
                var want: f64 = 0;
                for (lo[i]..np[i]) |p| want += s[p] / sum * vf[p][d];
                try std.testing.expectApproxEqAbs(want, out[i][d], 1e-4);
            }
        }
    }
}

test "tiled online-softmax attention with NQ=8 token tiling across sub-blocks" {
    var prng = std.Random.DefaultPrng.init(777);
    const r = prng.random();
    const hd = 64;
    const npos = 45;
    const ntiles = (npos + kv_tile - 1) / kv_tile;
    var kt: [ntiles * hd * kv_tile]f16 = undefined;
    var vr: [ntiles * kv_tile * hd]f16 = undefined;
    var kf: [npos][hd]f32 = undefined;
    var vf: [npos][hd]f32 = undefined;
    for (&kt) |*x| x.* = std.math.nan(f16);
    for (0..npos) |p| for (0..hd) |d| {
        const kk: f16 = @floatCast(r.floatNorm(f32));
        const vv: f16 = @floatCast(r.floatNorm(f32));
        kt[(p / kv_tile) * hd * kv_tile + d * kv_tile + p % kv_tile] = kk;
        vr[p * hd + d] = vv;
        kf[p][d] = kk;
        vf[p][d] = vv;
    };
    var q: [8][hd]f32 = undefined;
    for (&q) |*qq| for (qq) |*x| {
        x.* = r.floatNorm(f32) * 2;
    };
    const lo = [8]usize{ 10, 0, 5, 0, 15, 0, 7, 0 };
    const np = [8]usize{ npos, 25, 40, 16, npos, 10, 33, 4 };
    var out: [8][hd]f32 = undefined;
    const scale: f32 = 0.125;
    var qp: [8][*]const f32 = undefined;
    var outp: [8][*]f32 = undefined;
    for (0..8) |i| {
        qp[i] = &q[i];
        outp[i] = &out[i];
    }
    attendBlock(8, .f16, hd, @ptrCast(&kt), @ptrCast(&vr), qp, lo, np, outp, scale, 0, 0);
    for (0..8) |i| {
        var s: [npos]f64 = undefined;
        var mx: f64 = -std.math.inf(f64);
        for (lo[i]..np[i]) |p| {
            var dot: f64 = 0;
            for (0..hd) |d| dot += @as(f64, q[i][d]) * kf[p][d];
            s[p] = dot * scale;
            mx = @max(mx, s[p]);
        }
        var sum: f64 = 0;
        for (lo[i]..np[i]) |p| {
            s[p] = @exp(s[p] - mx);
            sum += s[p];
        }
        for (0..hd) |d| {
            var want: f64 = 0;
            for (lo[i]..np[i]) |p| want += s[p] / sum * vf[p][d];
            try std.testing.expectApproxEqAbs(want, out[i][d], 1e-4);
        }
    }

    // Also verify q4_0 with NQ=8
    const nblocks = hd / 32;
    const tile_bytes = hd * 9;
    var kt_q4: [ntiles * tile_bytes]u8 align(32) = undefined;
    var vr_q4: [ntiles * tile_bytes]u8 align(32) = undefined;
    @memset(&kt_q4, 0xFF);
    @memset(&vr_q4, 0xFF);
    for (0..npos) |p| {
        const tile = p / kv_tile;
        const pl = p % kv_tile;
        for (0..nblocks) |b| {
            var q4_k: [32]u8 = undefined;
            var sc_k: f16 = undefined;
            quantBlock32_q4(kf[p][b * 32 ..][0..32], &q4_k, &sc_k);

            const k_bp = kt_q4[tile * tile_bytes + b * 288 ..].ptr;
            if (pl == 0) @memset(k_bp[0..256], 0);
            @as(*align(1) f16, @ptrCast(k_bp + 256 + pl * 2)).* = sc_k;
            if (pl < 8) {
                inline for (0..32) |dp| {
                    k_bp[dp * 8 + pl] = (k_bp[dp * 8 + pl] & 0xF0) | q4_k[dp];
                }
            } else {
                inline for (0..32) |dp| {
                    k_bp[dp * 8 + (pl - 8)] = (k_bp[dp * 8 + (pl - 8)] & 0x0F) | (q4_k[dp] << 4);
                }
            }

            var q4_v: [32]u8 = undefined;
            var sc_v: f16 = undefined;
            quantBlock32_q4(vf[p][b * 32 ..][0..32], &q4_v, &sc_v);

            const vt = vr_q4[tile * tile_bytes ..].ptr;
            @as(*align(1) f16, @ptrCast(vt + 8 * hd + b * 32 + pl * 2)).* = sc_v;
            const v_dst = vt + pl * (hd / 2) + b * 16;
            const q_lo: @Vector(8, u8) = q4_v[0..8].*;
            const q_hi: @Vector(8, u8) = q4_v[8..16].*;
            const v0: [8]u8 = q_lo | (q_hi << @as(@Vector(8, u8), @splat(4)));
            const q_lo2: @Vector(8, u8) = q4_v[16..24].*;
            const q_hi2: @Vector(8, u8) = q4_v[24..32].*;
            const v1: [8]u8 = q_lo2 | (q_hi2 << @as(@Vector(8, u8), @splat(4)));
            @memcpy(v_dst[0..8], &v0);
            @memcpy(v_dst[8..16], &v1);
        }
    }
    var kf_q4: [npos][hd]f32 = undefined;
    var vf_q4: [npos][hd]f32 = undefined;
    for (0..npos) |p| {
        const tile = p / kv_tile;
        const pl = p % kv_tile;
        for (0..nblocks) |b| {
            const k_bp = kt_q4[tile * tile_bytes + b * 288 ..].ptr;
            const ks: f32 = @as(*align(1) f16, @ptrCast(k_bp + 256 + pl * 2)).*;
            for (0..32) |dp| {
                const u: u8 = if (pl < 8)
                    (k_bp[dp * 8 + pl] & 0x0F)
                else
                    ((k_bp[dp * 8 + (pl - 8)] >> 4) & 0x0F);
                const s: i8 = @as(i8, @bitCast(u ^ 8)) -% 8;
                kf_q4[p][b * 32 + dp] = @as(f32, @floatFromInt(s)) * ks;
            }

            const vt = vr_q4[tile * tile_bytes ..].ptr;
            const vs: f32 = @as(*align(1) f16, @ptrCast(vt + 8 * hd + b * 32 + pl * 2)).*;
            const row = vt + pl * (hd / 2) + b * 16;
            for (0..8) |j| {
                const v_lo = row[j] & 0x0F;
                const v_hi = (row[j] >> 4) & 0x0F;
                const s0: i8 = @as(i8, @bitCast(v_lo ^ 8)) -% 8;
                const s1: i8 = @as(i8, @bitCast(v_hi ^ 8)) -% 8;
                vf_q4[p][b * 32 + j] = @as(f32, @floatFromInt(s0)) * vs;
                vf_q4[p][b * 32 + 8 + j] = @as(f32, @floatFromInt(s1)) * vs;
            }
            for (0..8) |j| {
                const v_lo2 = row[8 + j] & 0x0F;
                const v_hi2 = (row[8 + j] >> 4) & 0x0F;
                const s2: i8 = @as(i8, @bitCast(v_lo2 ^ 8)) -% 8;
                const s3: i8 = @as(i8, @bitCast(v_hi2 ^ 8)) -% 8;
                vf_q4[p][b * 32 + 16 + j] = @as(f32, @floatFromInt(s2)) * vs;
                vf_q4[p][b * 32 + 24 + j] = @as(f32, @floatFromInt(s3)) * vs;
            }
        }
    }
    var out_q4: [8][hd]f32 = undefined;
    var outp_q4: [8][*]f32 = undefined;
    for (0..8) |i| outp_q4[i] = &out_q4[i];
    attendBlock(8, .q4_0, hd, &kt_q4, &vr_q4, qp, lo, np, outp_q4, scale, 0, 0);
    for (0..8) |i| {
        var s: [npos]f64 = undefined;
        var mx: f64 = -std.math.inf(f64);
        for (lo[i]..np[i]) |p| {
            var dot: f64 = 0;
            for (0..hd) |d| dot += @as(f64, q[i][d]) * kf_q4[p][d];
            s[p] = dot * scale;
            mx = @max(mx, s[p]);
        }
        var sum: f64 = 0;
        for (lo[i]..np[i]) |p| {
            s[p] = @exp(s[p] - mx);
            sum += s[p];
        }
        for (0..hd) |d| {
            var want: f64 = 0;
            for (lo[i]..np[i]) |p| want += s[p] / sum * vf_q4[p][d];
            try std.testing.expectApproxEqAbs(want, out_q4[i][d], 1e-4);
        }
    }

    // Also verify q2_0 with NQ=8
    const tile_bytes_q2 = hd * 5;
    var kt_q2: [ntiles * tile_bytes_q2]u8 align(32) = undefined;
    var vr_q2: [ntiles * tile_bytes_q2]u8 align(32) = undefined;
    @memset(&kt_q2, 0);
    @memset(&vr_q2, 0);
    for (0..npos) |p| {
        const tile = p / kv_tile;
        const pl = p % kv_tile;
        const byte_idx = pl % 4;
        const shift: u3 = @intCast((pl / 4) * 2);
        const mask = ~(@as(u8, 0x03) << shift);

        for (0..nblocks) |b| {
            var q2_k: [32]u8 = undefined;
            var sc_k: f16 = undefined;
            quantBlock32_q2(kf[p][b * 32 ..][0..32], &q2_k, &sc_k);

            const k_bp = kt_q2[tile * tile_bytes_q2 + b * 160 ..].ptr;
            if (pl == 0) @memset(k_bp[0..128], 0);
            @as(*align(1) f16, @ptrCast(k_bp + 128 + pl * 2)).* = sc_k;
            inline for (0..32) |dp| {
                k_bp[dp * 4 + byte_idx] = (k_bp[dp * 4 + byte_idx] & mask) | (q2_k[dp] << shift);
            }

            var q2_v: [32]u8 = undefined;
            var sc_v: f16 = undefined;
            quantBlock32_q2(vf[p][b * 32 ..][0..32], &q2_v, &sc_v);

            const vt = vr_q2[tile * tile_bytes_q2 ..].ptr;
            @as(*align(1) f16, @ptrCast(vt + 4 * hd + b * 32 + pl * 2)).* = sc_v;
            const v_dst = vt + pl * (hd / 4) + b * 8;
            const d0: @Vector(8, u8) = q2_v[0..8].*;
            const d1: @Vector(8, u8) = q2_v[8..16].*;
            const d2: @Vector(8, u8) = q2_v[16..24].*;
            const d3: @Vector(8, u8) = q2_v[24..32].*;
            const v_packed: [8]u8 = d0 | (d1 << @as(@Vector(8, u8), @splat(2))) | (d2 << @as(@Vector(8, u8), @splat(4))) | (d3 << @as(@Vector(8, u8), @splat(6)));
            @memcpy(v_dst[0..8], &v_packed);
        }
    }
    var kf_q2: [npos][hd]f32 = undefined;
    var vf_q2: [npos][hd]f32 = undefined;
    for (0..npos) |p| {
        const tile = p / kv_tile;
        const pl = p % kv_tile;
        const byte_idx = pl % 4;
        const shift: u3 = @intCast((pl / 4) * 2);

        for (0..nblocks) |b| {
            const k_bp = kt_q2[tile * tile_bytes_q2 + b * 160 ..].ptr;
            const ks: f32 = @as(*align(1) f16, @ptrCast(k_bp + 128 + pl * 2)).*;
            for (0..32) |dp| {
                const u: u8 = (k_bp[dp * 4 + byte_idx] >> shift) & 0x03;
                const level: f32 = @as(f32, @floatFromInt(u)) * 2.0 - 3.0;
                kf_q2[p][b * 32 + dp] = level * ks;
            }

            const vt = vr_q2[tile * tile_bytes_q2 ..].ptr;
            const vs: f32 = @as(*align(1) f16, @ptrCast(vt + 4 * hd + b * 32 + pl * 2)).*;
            const row = vt + pl * (hd / 4) + b * 8;
            for (0..8) |j| {
                const byte = row[j];
                const d0: f32 = @as(f32, @floatFromInt((byte >> 0) & 0x03)) * 2.0 - 3.0;
                const d1: f32 = @as(f32, @floatFromInt((byte >> 2) & 0x03)) * 2.0 - 3.0;
                const d2: f32 = @as(f32, @floatFromInt((byte >> 4) & 0x03)) * 2.0 - 3.0;
                const d3: f32 = @as(f32, @floatFromInt((byte >> 6) & 0x03)) * 2.0 - 3.0;
                vf_q2[p][b * 32 + j] = d0 * vs;
                vf_q2[p][b * 32 + 8 + j] = d1 * vs;
                vf_q2[p][b * 32 + 16 + j] = d2 * vs;
                vf_q2[p][b * 32 + 24 + j] = d3 * vs;
            }
        }
    }
    var out_q2: [8][hd]f32 = undefined;
    var outp_q2: [8][*]f32 = undefined;
    for (0..8) |i| outp_q2[i] = &out_q2[i];
    attendBlock(8, .q2_0, hd, &kt_q2, &vr_q2, qp, lo, np, outp_q2, scale, 0, 0);
    for (0..8) |i| {
        var s: [npos]f64 = undefined;
        var mx: f64 = -std.math.inf(f64);
        for (lo[i]..np[i]) |p| {
            var dot: f64 = 0;
            for (0..hd) |d| dot += @as(f64, q[i][d]) * kf_q2[p][d];
            s[p] = dot * scale;
            mx = @max(mx, s[p]);
        }
        var sum: f64 = 0;
        for (lo[i]..np[i]) |p| {
            s[p] = @exp(s[p] - mx);
            sum += s[p];
        }
        for (0..hd) |d| {
            var want: f64 = 0;
            for (lo[i]..np[i]) |p| want += s[p] / sum * vf_q2[p][d];
            try std.testing.expectApproxEqAbs(want, out_q2[i][d], 1e-4);
        }
    }
}

test "SWA ring buffer attention matches linear reference" {
    var prng = std.Random.DefaultPrng.init(1234);
    const r = prng.random();
    const hd = 64;
    const window = 32;
    const w_tiles = (window + kv_tile - 1) / kv_tile + 1;
    const npos = 83;

    var kt: [w_tiles * hd * kv_tile]f16 = undefined;
    var vr: [w_tiles * kv_tile * hd]f16 = undefined;
    var k_ref: [npos][hd]f32 = undefined;
    var v_ref: [npos][hd]f32 = undefined;

    for (0..npos) |p| {
        const tile = (p / kv_tile) % w_tiles;
        const pl = p % kv_tile;
        for (0..hd) |d| {
            const kk: f16 = @floatCast(r.floatNorm(f32));
            const vv: f16 = @floatCast(r.floatNorm(f32));
            kt[tile * hd * kv_tile + d * kv_tile + pl] = kk;
            vr[tile * kv_tile * hd + pl * hd + d] = vv;
            k_ref[p][d] = kk;
            v_ref[p][d] = vv;
        }
    }

    var q: [3][hd]f32 = undefined;
    for (&q) |*qq| for (qq) |*x| {
        x.* = r.floatNorm(f32) * 2;
    };
    const np = [3]usize{ 81, 82, 83 };
    const lo = [3]usize{ 81 - window, 82 - window, 83 - window };
    var out: [3][hd]f32 = undefined;
    const scale: f32 = 0.125;
    attendBlock(3, .f16, hd, @ptrCast(&kt), @ptrCast(&vr), .{ &q[0], &q[1], &q[2] }, lo, np, .{ &out[0], &out[1], &out[2] }, scale, window, w_tiles);

    for (0..3) |i| {
        var s: [npos]f64 = undefined;
        var mx: f64 = -std.math.inf(f64);
        for (lo[i]..np[i]) |p| {
            var dot: f64 = 0;
            for (0..hd) |d| dot += @as(f64, q[i][d]) * k_ref[p][d];
            s[p] = dot * scale;
            mx = @max(mx, s[p]);
        }
        var sum: f64 = 0;
        for (lo[i]..np[i]) |p| {
            s[p] = @exp(s[p] - mx);
            sum += s[p];
        }
        for (0..hd) |d| {
            var want: f64 = 0;
            for (lo[i]..np[i]) |p| want += s[p] / sum * v_ref[p][d];
            try std.testing.expectApproxEqAbs(want, out[i][d], 1e-4);
        }
    }

    const nblocks = hd / 32;
    const tile_bytes = hd * 17;
    var kt_q: [w_tiles * tile_bytes]u8 align(32) = undefined;
    var vr_q: [w_tiles * tile_bytes]u8 align(32) = undefined;
    @memset(&kt_q, 0xFF);
    @memset(&vr_q, 0xFF);

    var ksrc: [npos][hd]f32 = undefined;
    var vsrc: [npos][hd]f32 = undefined;
    for (&ksrc) |*row| for (row) |*x| {
        x.* = r.floatNorm(f32);
    };
    for (&vsrc) |*row| for (row) |*x| {
        x.* = r.floatNorm(f32);
    };

    for (0..npos) |p| {
        const tile = (p / kv_tile) % w_tiles;
        const pl = p % kv_tile;
        for (0..nblocks) |b| {
            quantBlock32(ksrc[p][b * 32 ..][0..32], kt_q[tile * tile_bytes + b * 544 + pl ..].ptr, kv_tile, @ptrCast(&kt_q[tile * tile_bytes + b * 544 + 512 + pl * 2]));
            quantBlock32(vsrc[p][b * 32 ..][0..32], vr_q[tile * tile_bytes + pl * hd + b * 32 ..].ptr, 1, @ptrCast(&vr_q[tile * tile_bytes + 16 * hd + b * 32 + pl * 2]));
        }
    }

    var kf_q: [npos][hd]f32 = undefined;
    var vf_q: [npos][hd]f32 = undefined;
    for (lo[0]..np[2]) |p| {
        const tile = (p / kv_tile) % w_tiles;
        const pl = p % kv_tile;
        for (0..nblocks) |b| {
            const ks: f32 = @as(*align(1) f16, @ptrCast(&kt_q[tile * tile_bytes + b * 544 + 512 + pl * 2])).*;
            const vs: f32 = @as(*align(1) f16, @ptrCast(&vr_q[tile * tile_bytes + 16 * hd + b * 32 + pl * 2])).*;
            for (0..32) |dp| {
                kf_q[p][b * 32 + dp] = @as(f32, @floatFromInt(@as(i8, @bitCast(kt_q[tile * tile_bytes + b * 544 + dp * 16 + pl])))) * ks;
                vf_q[p][b * 32 + dp] = @as(f32, @floatFromInt(@as(i8, @bitCast(vr_q[tile * tile_bytes + pl * hd + b * 32 + dp])))) * vs;
            }
        }
    }

    var out_q: [3][hd]f32 = undefined;
    attendBlock(3, .q8_0, hd, &kt_q, &vr_q, .{ &q[0], &q[1], &q[2] }, lo, np, .{ &out_q[0], &out_q[1], &out_q[2] }, scale, window, w_tiles);

    for (0..3) |i| {
        var s: [npos]f64 = undefined;
        var mx: f64 = -std.math.inf(f64);
        for (lo[i]..np[i]) |p| {
            var dot: f64 = 0;
            for (0..hd) |d| dot += @as(f64, q[i][d]) * kf_q[p][d];
            s[p] = dot * scale;
            mx = @max(mx, s[p]);
        }
        var sum: f64 = 0;
        for (lo[i]..np[i]) |p| {
            s[p] = @exp(s[p] - mx);
            sum += s[p];
        }
        for (0..hd) |d| {
            var want: f64 = 0;
            for (lo[i]..np[i]) |p| want += s[p] / sum * vf_q[p][d];
            try std.testing.expectApproxEqAbs(want, out_q[i][d], 1e-4);
        }
    }
}

test "FlashDecoding chunked attention matches unchunked reference across cache formats" {
    var prng = std.Random.DefaultPrng.init(12345);
    const r = prng.random();
    const hd = 128;
    const npos = 128;
    const ntiles = (npos + kv_tile - 1) / kv_tile;

    inline for (.{ CacheType.f16, CacheType.q8_0, CacheType.q4_0, CacheType.q2_0, CacheType.q2_k }) |cache_type| {
        const k_tile_bytes: usize = switch (cache_type) {
            .f16 => hd * 32,
            .q8_0 => hd * 17,
            .q4_0 => hd * 9,
            .q2_0, .q2_k => hd * 5,
        };
        const v_tile_bytes: usize = switch (cache_type) {
            .f16 => hd * 32,
            .q8_0 => hd * 17,
            .q4_0, .q2_k => hd * 9,
            .q2_0 => hd * 5,
        };
        var kt_buf: [ntiles * (hd * 32)]u8 align(32) = undefined;
        var vr_buf: [ntiles * (hd * 32)]u8 align(32) = undefined;
        const kt = kt_buf[0 .. ntiles * k_tile_bytes];
        const vr = vr_buf[0 .. ntiles * v_tile_bytes];
        @memset(kt, 0);
        @memset(vr, 0);

        var ksrc: [npos][hd]f32 = undefined;
        var vsrc: [npos][hd]f32 = undefined;
        for (&ksrc) |*row| for (row) |*x| {
            x.* = r.floatNorm(f32);
        };
        for (&vsrc) |*row| for (row) |*x| {
            x.* = r.floatNorm(f32);
        };

        const nblocks = hd / 32;
        switch (cache_type) {
            .f16 => {
                const kt_f16: [*]f16 = @ptrCast(@alignCast(kt.ptr));
                const vr_f16: [*]f16 = @ptrCast(@alignCast(vr.ptr));
                for (0..npos) |p| {
                    const tile = p / kv_tile;
                    const pl = p % kv_tile;
                    for (0..hd) |d| {
                        kt_f16[tile * hd * kv_tile + d * kv_tile + pl] = @floatCast(ksrc[p][d]);
                        vr_f16[tile * hd * kv_tile + pl * hd + d] = @floatCast(vsrc[p][d]);
                    }
                }
            },
            .q8_0 => {
                for (0..npos) |p| {
                    const tile = p / kv_tile;
                    const pl = p % kv_tile;
                    for (0..nblocks) |b| {
                        quantBlock32(ksrc[p][b * 32 ..][0..32], kt.ptr + tile * k_tile_bytes + b * 544 + pl, kv_tile, @ptrCast(kt.ptr + tile * k_tile_bytes + b * 544 + 512 + pl * 2));
                        quantBlock32(vsrc[p][b * 32 ..][0..32], vr.ptr + tile * v_tile_bytes + pl * hd + b * 32, 1, @ptrCast(vr.ptr + tile * v_tile_bytes + 16 * hd + b * 32 + pl * 2));
                    }
                }
            },
            .q4_0 => {
                for (0..npos) |p| {
                    const tile = p / kv_tile;
                    const pl = p % kv_tile;
                    for (0..nblocks) |b| {
                        var q4_k: [32]u8 = undefined;
                        var sc_k: f16 = undefined;
                        quantBlock32_q4(ksrc[p][b * 32 ..][0..32], &q4_k, &sc_k);

                        const k_bp = kt.ptr + tile * k_tile_bytes + b * 288;
                        if (pl == 0) @memset(k_bp[0..256], 0);
                        @as(*align(1) f16, @ptrCast(k_bp + 256 + pl * 2)).* = sc_k;
                        if (pl < 8) {
                            inline for (0..32) |dp| {
                                k_bp[dp * 8 + pl] = (k_bp[dp * 8 + pl] & 0xF0) | q4_k[dp];
                            }
                        } else {
                            inline for (0..32) |dp| {
                                k_bp[dp * 8 + (pl - 8)] = (k_bp[dp * 8 + (pl - 8)] & 0x0F) | (q4_k[dp] << 4);
                            }
                        }

                        var q4_v: [32]u8 = undefined;
                        var sc_v: f16 = undefined;
                        quantBlock32_q4(vsrc[p][b * 32 ..][0..32], &q4_v, &sc_v);

                        const vt = vr.ptr + tile * v_tile_bytes;
                        @as(*align(1) f16, @ptrCast(vt + 8 * hd + b * 32 + pl * 2)).* = sc_v;
                        const v_dst = vt + pl * (hd / 2) + b * 16;
                        const q_lo: @Vector(8, u8) = q4_v[0..8].*;
                        const q_hi: @Vector(8, u8) = q4_v[8..16].*;
                        const v0: [8]u8 = q_lo | (q_hi << @as(@Vector(8, u8), @splat(4)));
                        const q_lo2: @Vector(8, u8) = q4_v[16..24].*;
                        const q_hi2: @Vector(8, u8) = q4_v[24..32].*;
                        const v1: [8]u8 = q_lo2 | (q_hi2 << @as(@Vector(8, u8), @splat(4)));
                        @memcpy(v_dst[0..8], &v0);
                        @memcpy(v_dst[8..16], &v1);
                    }
                }
            },
            .q2_0 => {
                for (0..npos) |p| {
                    const tile = p / kv_tile;
                    const pl = p % kv_tile;
                    const byte_idx = pl % 4;
                    const shift: u3 = @intCast((pl / 4) * 2);
                    const mask = ~(@as(u8, 0x03) << shift);

                    for (0..nblocks) |b| {
                        var q2_k: [32]u8 = undefined;
                        var sc_k: f16 = undefined;
                        quantBlock32_q2(ksrc[p][b * 32 ..][0..32], &q2_k, &sc_k);

                        const k_bp = kt.ptr + tile * k_tile_bytes + b * 160;
                        if (pl == 0) @memset(k_bp[0..128], 0);
                        @as(*align(1) f16, @ptrCast(k_bp + 128 + pl * 2)).* = sc_k;
                        inline for (0..32) |dp| {
                            k_bp[dp * 4 + byte_idx] = (k_bp[dp * 4 + byte_idx] & mask) | (q2_k[dp] << shift);
                        }

                        var q2_v: [32]u8 = undefined;
                        var sc_v: f16 = undefined;
                        quantBlock32_q2(vsrc[p][b * 32 ..][0..32], &q2_v, &sc_v);

                        const vt = vr.ptr + tile * v_tile_bytes;
                        @as(*align(1) f16, @ptrCast(vt + 4 * hd + b * 32 + pl * 2)).* = sc_v;
                        const v_dst = vt + pl * (hd / 4) + b * 8;
                        const d0: @Vector(8, u8) = q2_v[0..8].*;
                        const d1: @Vector(8, u8) = q2_v[8..16].*;
                        const d2: @Vector(8, u8) = q2_v[16..24].*;
                        const d3: @Vector(8, u8) = q2_v[24..32].*;
                        const v_packed: [8]u8 = d0 | (d1 << @as(@Vector(8, u8), @splat(2))) | (d2 << @as(@Vector(8, u8), @splat(4))) | (d3 << @as(@Vector(8, u8), @splat(6)));
                        @memcpy(v_dst[0..8], &v_packed);
                    }
                }
            },
            .q2_k => {
                for (0..npos) |p| {
                    const tile = p / kv_tile;
                    const pl = p % kv_tile;
                    const byte_idx = pl % 4;
                    const shift: u3 = @intCast((pl / 4) * 2);
                    const mask = ~(@as(u8, 0x03) << shift);

                    for (0..nblocks) |b| {
                        var q2_k: [32]u8 = undefined;
                        var sc_k: f16 = undefined;
                        quantBlock32_q2(ksrc[p][b * 32 ..][0..32], &q2_k, &sc_k);

                        const k_bp = kt.ptr + tile * k_tile_bytes + b * 160;
                        if (pl == 0) @memset(k_bp[0..128], 0);
                        @as(*align(1) f16, @ptrCast(k_bp + 128 + pl * 2)).* = sc_k;
                        inline for (0..32) |dp| {
                            k_bp[dp * 4 + byte_idx] = (k_bp[dp * 4 + byte_idx] & mask) | (q2_k[dp] << shift);
                        }

                        var q4_v: [32]u8 = undefined;
                        var sc_v: f16 = undefined;
                        quantBlock32_q4(vsrc[p][b * 32 ..][0..32], &q4_v, &sc_v);

                        const vt = vr.ptr + tile * v_tile_bytes;
                        @as(*align(1) f16, @ptrCast(vt + 8 * hd + b * 32 + pl * 2)).* = sc_v;
                        const v_dst = vt + pl * (hd / 2) + b * 16;
                        const q_lo: @Vector(8, u8) = q4_v[0..8].*;
                        const q_hi: @Vector(8, u8) = q4_v[8..16].*;
                        const v0: [8]u8 = q_lo | (q_hi << @as(@Vector(8, u8), @splat(4)));
                        const q_lo2: @Vector(8, u8) = q4_v[16..24].*;
                        const q_hi2: @Vector(8, u8) = q4_v[24..32].*;
                        const v1: [8]u8 = q_lo2 | (q_hi2 << @as(@Vector(8, u8), @splat(4)));
                        @memcpy(v_dst[0..8], &v0);
                        @memcpy(v_dst[8..16], &v1);
                    }
                }
            },
        }

        var q: [hd]f32 = undefined;
        for (&q) |*x| {
            x.* = r.floatNorm(f32) * 2;
        }
        const lo = [1]usize{0};
        const np = [1]usize{npos};
        var ref_out: [1][hd]f32 = undefined;
        const scale: f32 = 0.125;
        attendBlock(1, cache_type, hd, kt.ptr, vr.ptr, .{&q}, lo, np, .{&ref_out[0]}, scale, 0, 0);

        inline for (.{ 2, 4, 8 }) |n_chunks| {
            var chunk_out: [n_chunks][hd]f32 = undefined;
            var ps: [n_chunks]PartialStats = undefined;
            const chunk_tiles = (ntiles + n_chunks - 1) / n_chunks;
            for (0..n_chunks) |c_idx| {
                const t_start = c_idx * chunk_tiles;
                const t_end = t_start + chunk_tiles;
                attendBlockChunk(1, cache_type, hd, kt.ptr, vr.ptr, .{&q}, lo, np, .{&chunk_out[c_idx]}, scale, 0, 0, t_start, t_end, &ps[c_idx]);
            }
            var max_m: f32 = -std.math.inf(f32);
            for (0..n_chunks) |c_idx| {
                max_m = @max(max_m, ps[c_idx].m[0]);
            }
            var total_l: f32 = 0;
            var alpha: [n_chunks]f32 = undefined;
            for (0..n_chunks) |c_idx| {
                if (ps[c_idx].m[0] > -std.math.inf(f32)) {
                    const a = @exp(ps[c_idx].m[0] - max_m);
                    alpha[c_idx] = a;
                    total_l += ps[c_idx].l[0] * a;
                } else {
                    alpha[c_idx] = 0;
                }
            }
            const inv_l = if (total_l > 0) 1.0 / total_l else 0.0;
            for (0..hd) |d| {
                var acc: f32 = 0;
                for (0..n_chunks) |c_idx| {
                    if (alpha[c_idx] > 0) acc += alpha[c_idx] * chunk_out[c_idx][d];
                }
                const got = acc * inv_l;
                try std.testing.expectApproxEqAbs(ref_out[0][d], got, 1e-4);
            }
        }
    }
}

test "PMM adaptive scale and asymmetric shift estimator on symmetric vs skewed data" {
    var sym_xs: [32]f32 = undefined;
    for (0..16) |i| {
        const val: f32 = @as(f32, @floatFromInt(i + 1)) * 0.1;
        sym_xs[i] = val;
        sym_xs[i + 16] = -val;
    }
    const sym_amax: f32 = 1.6;
    const s_q4 = pmmScaleBlock32_q4(&sym_xs, sym_amax);
    const s_q2 = pmmScaleBlock32_q2(&sym_xs, sym_amax);
    try std.testing.expect(s_q4 > 1.45 and s_q4 <= sym_amax);
    try std.testing.expect(s_q2 > 1.25 and s_q2 <= sym_amax);

    var skewed_xs: [32]f32 = undefined;
    for (0..31) |i| {
        skewed_xs[i] = @as(f32, @floatFromInt(i % 5)) * 0.2 - 0.2;
    }
    skewed_xs[31] = 6.0;
    const skew_amax: f32 = 6.0;

    const skew_s_q4 = pmmScaleBlock32_q4(&skewed_xs, skew_amax);
    const skew_s_q2 = pmmScaleBlock32_q2(&skewed_xs, skew_amax);

    const ratio_q4 = skew_s_q4 / skew_amax;
    const ratio_q2 = skew_s_q2 / skew_amax;
    try std.testing.expect(ratio_q4 < 0.95);
    try std.testing.expect(ratio_q2 < 0.85);

    var q2: [32]u8 = undefined;
    var sc_q2: f16 = undefined;
    quantBlock32_q2(&skewed_xs, &q2, &sc_q2);
    try std.testing.expect(sc_q2 > 0);

    var q4: [32]u8 = undefined;
    var sc_q4: f16 = undefined;
    quantBlock32_q4(&skewed_xs, &q4, &sc_q4);
    try std.testing.expect(sc_q4 > 0);
}

/// DeepSeek Multi-Head Latent Attention (MLA) decoupled keys & latent projection.
pub const MlaParams = struct {
    kv_lora_rank: usize = 512,      // d_c
    qk_rope_head_dim: usize = 64,   // d_R
    qk_nope_head_dim: usize = 128,  // d_nope
    v_head_dim: usize = 128,        // d_v
    n_heads: usize = 16,
};

/// Projects and splits input into latent KV vector c_kv [kv_lora_rank] and decoupled position key k_rope [qk_rope_head_dim].
pub fn mlaSplitKv(
    raw_kv: []const f32,
    c_kv: []f32,
    k_rope: []f32,
    params: MlaParams,
) void {
    const dc = params.kv_lora_rank;
    const dr = params.qk_rope_head_dim;
    @memcpy(c_kv[0..dc], raw_kv[0..dc]);
    @memcpy(k_rope[0..dr], raw_kv[dc .. dc + dr]);
}

/// Computes MLA attention score combining content dot-product and decoupled RoPE dot-product:
/// score = <q_nope, k_nope> + <q_rope, k_rope>
pub fn mlaScore(
    q_nope: []const f32,
    k_nope: []const f32,
    q_rope: []const f32,
    k_rope: []const f32,
) f32 {
    var score: f32 = 0.0;
    var i: usize = 0;
    while (i + 8 <= q_nope.len) : (i += 8) {
        const qv: V8f = q_nope[i..][0..8].*;
        const kv: V8f = k_nope[i..][0..8].*;
        score += @reduce(.Add, qv * kv);
    }
    while (i < q_nope.len) : (i += 1) {
        score += q_nope[i] * k_nope[i];
    }
    var j: usize = 0;
    while (j + 8 <= q_rope.len) : (j += 8) {
        const qv: V8f = q_rope[j..][0..8].*;
        const kv: V8f = k_rope[j..][0..8].*;
        score += @reduce(.Add, qv * kv);
    }
    while (j < q_rope.len) : (j += 1) {
        score += q_rope[j] * k_rope[j];
    }
    return score;
}

test "DeepSeek MLA split and score computation" {
    const params = MlaParams{
        .kv_lora_rank = 16,
        .qk_rope_head_dim = 8,
        .qk_nope_head_dim = 16,
        .v_head_dim = 16,
        .n_heads = 2,
    };
    var raw: [24]f32 = undefined;
    for (0..24) |i| raw[i] = @floatFromInt(i + 1);

    var c_kv: [16]f32 = undefined;
    var k_rope: [8]f32 = undefined;
    mlaSplitKv(&raw, &c_kv, &k_rope, params);
    try std.testing.expectEqual(1.0, c_kv[0]);
    try std.testing.expectEqual(16.0, c_kv[15]);
    try std.testing.expectEqual(17.0, k_rope[0]);
    try std.testing.expectEqual(24.0, k_rope[7]);

    var q_nope: [16]f32 = undefined;
    var q_rope: [8]f32 = undefined;
    @memset(&q_nope, 1.0);
    @memset(&q_rope, 0.5);

    const s = mlaScore(&q_nope, &c_kv, &q_rope, &k_rope);
    // <1.0, 1..16> = 136.0
    // <0.5, 17..24> = 0.5 * 164 = 82.0
    // Total = 136 + 82 = 218.0
    try std.testing.expectApproxEqAbs(@as(f32, 218.0), s, 1e-4);
}
