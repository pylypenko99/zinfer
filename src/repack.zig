//! Row-interleaved ("x8") weight layouts and their GEMM kernels.
//!
//! At load time every quantized matmul weight is regrouped so that one 32-byte
//! vector holds 4 consecutive weights from each of 8 rows. A 4-byte slice of
//! the activation is broadcast to all lanes, so vpmaddubsw produces partial
//! sums for 8 output rows at once and the vector lanes *are* the rows: no
//! horizontal reductions, scales/mins apply to 8 rows per instruction, and
//! small-range formats accumulate several groups in i16 before widening.
//! The quantized bits are preserved exactly; only their order changes.
const std = @import("std");
const quant = @import("quant.zig");
const GgmlType = @import("gguf.zig").GgmlType;
const Q8K = quant.Q8K;
const Q8B = quant.Q8B;
const V32u8 = quant.V32u8;
const V32i8 = quant.V32i8;
const V16i16 = quant.V16i16;
const V8i32 = quant.V8i32;
const V8f = quant.V8f;
const maddubs = quant.maddubs;
const maddwd = quant.maddwd;
const toF = quant.toF;

extern fn @"llvm.x86.avx2.psign.b"(@Vector(32, i8), @Vector(32, i8)) @Vector(32, i8);

pub fn supported(t: GgmlType) bool {
    return switch (t) {
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q4_k, .q5_k, .q6_k => true,
        else => false,
    };
}

/// Bytes of one x8 block (8 rows x one source block).
pub fn blockBytes(t: GgmlType) usize {
    return switch (t) {
        .q4_0 => 144, // d[8]f16 | qs[4 k][8 r][4]
        .q4_1 => 160, // d[8] | m[8] | qs[4 k][8 r][4]
        .q5_0 => 176, // d[8] | qh[4 k][8 r] | qs[4 k][8 r][4]
        .q5_1 => 192, // d[8] | m[8] | qh[4 k][8 r] | qs[4 k][8 r][4]
        .q8_0 => 272, // d[8]f16 | qs[8 k][8 r][4]
        .q4_k => 1184, // d[8] | dmin[8] | sc[8 j][8 r] | mn[8 j][8 r] | qs[4 c][8 k][8 r][4]
        .q5_k => 1440, // q4_k layout with qh[8 k][8 r][4] inserted before qs
        .q6_k => 1696, // d[8] | pad 16 | sc[16 s][8 r] | {ql_a, ql_b, qh}[2 n][8 k][8 r][4]
        else => unreachable,
    };
}

/// Bytes of 8 rows of `cols` elements.
pub fn groupBytes(t: GgmlType, cols: usize) usize {
    return cols / t.layout().?.block * blockBytes(t);
}

// ---------------------------------------------------------------------------
// Repacking

/// Interleave 8 source rows (each `nb` blocks of type `t`) into `dst`.
pub fn repackGroup(t: GgmlType, rows: [8][*]const u8, nb: usize, dst: [*]u8) void {
    const bb = blockBytes(t);
    const sb = t.layout().?.bytes;
    for (0..nb) |b| {
        const o = dst + b * bb;
        for (rows, 0..) |row, r| {
            const x = row + b * sb;
            switch (t) {
                .q4_0, .q8_0 => {
                    @memcpy(o[r * 2 ..][0..2], x[0..2]);
                    const ng: usize = if (t == .q4_0) 4 else 8;
                    for (0..ng) |k| @memcpy(o[16 + (k * 8 + r) * 4 ..][0..4], x[2 + 4 * k ..][0..4]);
                },
                .q4_1, .q5_0, .q5_1 => {
                    const has_m = t != .q5_0;
                    const has_h = t != .q4_1;
                    @memcpy(o[r * 2 ..][0..2], x[0..2]);
                    if (has_m) @memcpy(o[16 + r * 2 ..][0..2], x[2..4]);
                    const qh_src: usize = if (has_m) 4 else 2;
                    const qh_off: usize = if (has_m) 32 else 16;
                    const qs_src = qh_src + @as(usize, if (has_h) 4 else 0);
                    const qs_off = qh_off + @as(usize, if (has_h) 32 else 0);
                    if (has_h) {
                        // Byte (k, r): the 5th bits of row r's values 4k..4k+3
                        // (low nibble) and 16+4k..16+4k+3 (high nibble).
                        const qh = std.mem.readInt(u32, x[qh_src..][0..4], .little);
                        for (0..4) |k| o[qh_off + k * 8 + r] = @truncate(((qh >> @intCast(4 * k)) & 0xf) | ((qh >> @intCast(16 + 4 * k)) & 0xf) << 4);
                    }
                    for (0..4) |k| @memcpy(o[qs_off + (k * 8 + r) * 4 ..][0..4], x[qs_src + 4 * k ..][0..4]);
                },
                .q4_k, .q5_k => {
                    @memcpy(o[r * 2 ..][0..2], x[0..2]);
                    @memcpy(o[16 + r * 2 ..][0..2], x[2..4]);
                    const sm = quant.scalesK4(x[4..16]);
                    for (0..8) |j| {
                        o[32 + j * 8 + r] = sm[j];
                        o[96 + j * 8 + r] = sm[8 + j];
                    }
                    const is5 = t == .q5_k;
                    const qs = if (is5) x + 48 else x + 16;
                    const qs_off: usize = if (is5) 416 else 160;
                    if (is5) for (0..8) |k| @memcpy(o[160 + (k * 8 + r) * 4 ..][0..4], x[16 + 4 * k ..][0..4]);
                    for (0..4) |c| for (0..8) |k| {
                        @memcpy(o[qs_off + ((c * 8 + k) * 8 + r) * 4 ..][0..4], qs[c * 32 + 4 * k ..][0..4]);
                    };
                },
                .q6_k => {
                    @memcpy(o[r * 2 ..][0..2], x[208..210]);
                    for (0..16) |s| o[32 + s * 8 + r] = x[192 + s];
                    for (0..2) |n| for (0..8) |k| {
                        const base = o + 160 + (n * 8 + k) * 96 + r * 4;
                        @memcpy(base[0..4], x[64 * n + 4 * k ..][0..4]);
                        @memcpy(base[32..36], x[64 * n + 32 + 4 * k ..][0..4]);
                        @memcpy(base[64..68], x[128 + 32 * n + 4 * k ..][0..4]);
                    };
                },
                else => unreachable,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel helpers

inline fn load32(p: [*]const u8) V32u8 {
    return p[0..32].*;
}
/// Unaligned vector load. Bitcasting a `[N]u8` copy instead makes Zig 0.17's
/// LLVM assemble the value byte by byte (~30 instructions per load).
inline fn loadVec(comptime V: type, p: [*]const u8) V {
    return @as(*align(1) const V, @ptrCast(p)).*;
}

/// Broadcast 4 bytes to all 8 lanes (vpbroadcastd from memory).
inline fn bc4(p: [*]const i8) V32i8 {
    const v: @Vector(8, u32) = @splat(@as(*align(1) const u32, @ptrCast(p)).*);
    return @bitCast(v);
}

/// 8 fp16 values -> 8 floats.
inline fn halfs(p: [*]const u8) V8f {
    const h = loadVec(@Vector(8, f16), p);
    return @floatCast(h);
}

const dup_mask: [16]i32 = .{ 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7 };
const pair_mask: [16]i32 = .{ 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15 };

/// 8 per-row bytes -> i16 lanes (r, r) for vpmaddwd against row pair sums.
inline fn dupU8(p: [*]const u8) V16i16 {
    const v: @Vector(8, u8) = p[0..8].*;
    return @intCast(@shuffle(u8, v, undefined, dup_mask));
}

inline fn dupI8(p: [*]const u8) V16i16 {
    const v = loadVec(@Vector(8, i8), p);
    return @intCast(@shuffle(i8, v, undefined, dup_mask));
}

/// Two 8-byte per-row arrays a, b -> i16 lanes (a[r], b[r]).
inline fn pairU8(p: [*]const u8) V16i16 {
    const v: @Vector(16, u8) = p[0..16].*;
    return @intCast(@shuffle(u8, v, undefined, pair_mask));
}

inline fn pairI8(p: [*]const u8) V16i16 {
    const v = loadVec(@Vector(16, i8), p);
    return @intCast(@shuffle(i8, v, undefined, pair_mask));
}

/// a * sign(s): negate where s < 0 (vpsignb; zero lanes are irrelevant here).
inline fn applySign(a: V32i8, s: V32i8) V32i8 {
    if (quant.has_avx2) return @"llvm.x86.avx2.psign.b"(a, s);
    return @select(i8, s < @as(V32i8, @splat(0)), @as(V32i8, @splat(0)) -% a, a);
}

/// One byte per row (8 rows) -> lanes 4r..4r+3 of row r.
inline fn spreadRows(p: [*]const u8) V32u8 {
    const v: @Vector(8, u8) = p[0..8].*;
    return @shuffle(u8, v, undefined, [32]i32{ 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7 });
}

/// Bits of a spreadRows byte holding lane i's 5th bit: low-nibble value i,
/// high-nibble value i.
const h_lo: V32u8 = @bitCast(@as(@Vector(8, u32), @splat(0x08040201)));
const h_hi: V32u8 = @bitCast(@as(@Vector(8, u32), @splat(0x80402010)));

const ones16: V16i16 = @splat(1);
const m4: V32u8 = @splat(0x0f);

/// Hide a constant from the optimizer. Used as a loop bound so LLVM keeps the
/// group loop rolled: fully unrolled, it interleaves token-major, unpacks every
/// weight vector up front and spills them, roughly doubling load traffic.
inline fn hidden(n: usize) usize {
    var v = n;
    std.mem.doNotOptimizeAway(&v);
    return v;
}

/// Bytes ahead of the current block to prefetch. Hardware prefetchers stop at
/// 4 KiB page boundaries; streaming weights at full DRAM bandwidth during
/// generation needs a few KiB of explicit lookahead per thread.
const prefetch_distance = 4096;

/// Prefetch the cache lines `prefetch_distance` ahead of one block.
inline fn prefetchAhead(blk: [*]const u8, comptime block_bytes: usize) void {
    inline for (0..(block_bytes + 63) / 64) |l| @prefetch(blk + prefetch_distance + l * 64, .{});
}
// ---------------------------------------------------------------------------
// Kernels: one 8-row group (nb x8 blocks) against NT activation rows.
// Result lane r is row r.

fn gemvK45(comptime is5: bool, w: [*]const u8, act: [*]const Q8K, nb: usize) V8f {
    @setEvalBranchQuota(100_000);
    const BS = if (is5) 1440 else 1184;
    const qs_off = if (is5) 416 else 160;
    const flush_every = if (is5) 4 else 8;
    var out: V8f = @splat(0);
    for (0..nb) |b| {
        const blk = w + b * BS;
        prefetchAhead(blk, BS);
        const y = &act[b];
        var acc: V8i32 = @splat(0);
        inline for (0..4) |c| {
            const scl = dupU8(blk + 32 + (2 * c) * 8);
            const sch = dupU8(blk + 32 + (2 * c + 1) * 8);
            const mlo: V32u8 = @splat(@as(u8, 1) << @intCast(2 * c));
            const mhi: V32u8 = @splat(@as(u8, 2) << @intCast(2 * c));
            inline for (0..8 / flush_every) |f| {
                var slo0: V16i16 = @splat(0);
                var slo1: V16i16 = @splat(0);
                var shi0: V16i16 = @splat(0);
                var shi1: V16i16 = @splat(0);
                inline for (0..flush_every / 2) |pair| {
                    const k0 = f * flush_every + pair * 2;
                    const k1 = k0 + 1;

                    const wv0 = load32(blk + qs_off + (c * 8 + k0) * 32);
                    const wv1 = load32(blk + qs_off + (c * 8 + k1) * 32);
                    var lo0 = wv0 & m4;
                    var hi0 = wv0 >> @splat(4);
                    var lo1 = wv1 & m4;
                    var hi1 = wv1 >> @splat(4);

                    if (is5) {
                        const sixteen: V32u8 = @splat(16);
                        const zero: V32u8 = @splat(0);
                        const h0 = load32(blk + 160 + k0 * 32);
                        const h1 = load32(blk + 160 + k1 * 32);
                        lo0 |= @select(u8, (h0 & mlo) == mlo, sixteen, zero);
                        hi0 |= @select(u8, (h0 & mhi) == mhi, sixteen, zero);
                        lo1 |= @select(u8, (h1 & mlo) == mlo, sixteen, zero);
                        hi1 |= @select(u8, (h1 & mhi) == mhi, sixteen, zero);
                    }

                    slo0 +%= maddubs(lo0, bc4(y.qs[64 * c + 4 * k0 ..].ptr));
                    shi0 +%= maddubs(hi0, bc4(y.qs[64 * c + 32 + 4 * k0 ..].ptr));
                    slo1 +%= maddubs(lo1, bc4(y.qs[64 * c + 4 * k1 ..].ptr));
                    shi1 +%= maddubs(hi1, bc4(y.qs[64 * c + 32 + 4 * k1 ..].ptr));
                }
                const slo = slo0 +% slo1;
                const shi = shi0 +% shi1;
                acc += maddwd(slo, scl) + maddwd(shi, sch);
            }
        }
        var mm: V8i32 = @splat(0);
        inline for (0..4) |p| {
            const mp = pairU8(blk + 96 + 2 * p * 8);
            mm += maddwd(mp, @bitCast(bc4(@ptrCast(&y.bsums32[2 * p]))));
        }
        const d8 = halfs(blk);
        const dmin8 = halfs(blk + 16);
        const yd: V8f = @splat(y.d);
        out = @mulAdd(V8f, d8 * yd, toF(acc), out);
        out = @mulAdd(V8f, -dmin8 * yd, toF(mm), out);
    }
    return out;
}

fn gemmK45(comptime is5: bool, comptime NT: usize, w: [*]const u8, acts: [NT][*]const Q8K, nb: usize) [NT]V8f {
    @setEvalBranchQuota(100_000);
    const BS = if (is5) 1440 else 1184;
    const qs_off = if (is5) 416 else 160;
    // Q4 values (<=15) allow 8 i16 accumulations per lane, Q5 (<=31) only 4.
    const flush_every = if (is5) 4 else 8;
    var out: [NT]V8f = @splat(@splat(0));
    const n_chunks = hidden(4);
    const n_k = hidden(flush_every);
    for (0..nb) |b| {
        const blk = w + b * BS;
        prefetchAhead(blk, BS);
        var acc: [NT]V8i32 = @splat(@splat(0));
        var c: usize = 0;
        while (c < n_chunks) : (c += 1) {
            const scl = dupU8(blk + 32 + (2 * c) * 8);
            const sch = dupU8(blk + 32 + (2 * c + 1) * 8);
            // Q5: bit 2c of qh is the 5th bit of the low nibbles, 2c+1 of the high.
            const mlo: V32u8 = @splat(@as(u8, 1) << @intCast(2 * c));
            const mhi: V32u8 = @splat(@as(u8, 2) << @intCast(2 * c));
            inline for (0..8 / flush_every) |f| {
                var slo: [NT]V16i16 = @splat(@splat(0));
                var shi: [NT]V16i16 = @splat(@splat(0));
                var kk: usize = 0;
                while (kk < n_k) : (kk += 1) {
                    const k = f * flush_every + kk;
                    const wv = load32(blk + qs_off + (c * 8 + k) * 32);
                    var lo = wv & m4;
                    var hi = wv >> @splat(4);
                    if (is5) {
                        const h = load32(blk + 160 + k * 32);
                        const sixteen: V32u8 = @splat(16);
                        const zero: V32u8 = @splat(0);
                        lo |= @select(u8, (h & mlo) == mlo, sixteen, zero);
                        hi |= @select(u8, (h & mhi) == mhi, sixteen, zero);
                    }
                    inline for (0..NT) |t| {
                        const y = &acts[t][b];
                        slo[t] +%= maddubs(lo, bc4(y.qs[64 * c + 4 * k ..].ptr));
                        shi[t] +%= maddubs(hi, bc4(y.qs[64 * c + 32 + 4 * k ..].ptr));
                    }
                }
                inline for (0..NT) |t| acc[t] += maddwd(slo[t], scl) + maddwd(shi[t], sch);
            }
        }
        // mins: sum_j mn[j][r] * bsums32[j], two sub-blocks per vpmaddwd
        var mm: [NT]V8i32 = @splat(@splat(0));
        inline for (0..4) |p| {
            const mp = pairU8(blk + 96 + 2 * p * 8);
            inline for (0..NT) |t| mm[t] += maddwd(mp, @bitCast(bc4(@ptrCast(&acts[t][b].bsums32[2 * p]))));
        }
        const d8 = halfs(blk);
        const dmin8 = halfs(blk + 16);
        inline for (0..NT) |t| {
            const yd: V8f = @splat(acts[t][b].d);
            out[t] = @mulAdd(V8f, d8 * yd, toF(acc[t]), out[t]);
            out[t] = @mulAdd(V8f, -dmin8 * yd, toF(mm[t]), out[t]);
        }
    }
    return out;
}

/// 6-bit value for output block m (0..3) of one 4-element group.
inline fn q6(comptime m: usize, la: V32u8, lb: V32u8, h: V32u8) V32u8 {
    const m3: V32u8 = @splat(3);
    return switch (m) {
        0 => (la & m4) | ((h & m3) << @splat(4)),
        1 => (lb & m4) | (((h >> @splat(2)) & m3) << @splat(4)),
        2 => (la >> @splat(4)) | (((h >> @splat(4)) & m3) << @splat(4)),
        3 => (lb >> @splat(4)) | ((h >> @splat(6)) << @splat(4)),
        else => unreachable,
    };
}

fn gemvQ6K(w: [*]const u8, act: [*]const Q8K, nb: usize) V8f {
    @setEvalBranchQuota(100_000);
    var out: V8f = @splat(0);
    for (0..nb) |b| {
        const blk = w + b * 1696;
        prefetchAhead(blk, 1696);
        const y = &act[b];
        var acc: V8i32 = @splat(0);
        inline for (0..8) |p| {
            const sp = pairI8(blk + 32 + 2 * p * 8);
            acc -%= maddwd(sp, @bitCast(bc4(@ptrCast(&y.bsums[2 * p])))) << @splat(5);
        }
        inline for (0..2) |n| {
            inline for (0..2) |kh| {
                var sc: [4]V16i16 = undefined;
                inline for (0..4) |m| sc[m] = dupI8(blk + 32 + (8 * n + 2 * m + kh) * 8);
                inline for (0..2) |kp| {
                    const k = kh * 4 + 2 * kp;
                    const b0 = blk + 160 + (n * 8 + k) * 96;
                    const la0 = load32(b0);
                    const lb0 = load32(b0 + 32);
                    const h0 = load32(b0 + 64);
                    const la1 = load32(b0 + 96);
                    const lb1 = load32(b0 + 128);
                    const h1 = load32(b0 + 160);
                    inline for (0..4) |m| {
                        const q0 = q6(m, la0, lb0, h0);
                        const q1 = q6(m, la1, lb1, h1);
                        const a = y.qs[128 * n + 32 * m + 4 * k ..];
                        acc += maddwd(maddubs(q0, bc4(a.ptr)) +% maddubs(q1, bc4(a.ptr + 4)), sc[m]);
                    }
                }
            }
        }
        const d8 = halfs(blk);
        out = @mulAdd(V8f, d8 * @as(V8f, @splat(y.d)), toF(acc), out);
    }
    return out;
}

fn gemmQ6K(comptime NT: usize, w: [*]const u8, acts: [NT][*]const Q8K, nb: usize) [NT]V8f {
    @setEvalBranchQuota(100_000);
    var out: [NT]V8f = @splat(@splat(0));
    const n_half = hidden(2);
    const n_pairs = hidden(2);
    for (0..nb) |b| {
        const blk = w + b * 1696;
        prefetchAhead(blk, 1696);
        var acc: [NT]V8i32 = @splat(@splat(0));
        // Stored values are q+32: subtract 32 * sum_s sc[s][r] * bsums[s].
        inline for (0..8) |p| {
            const sp = pairI8(blk + 32 + 2 * p * 8);
            inline for (0..NT) |t| acc[t] -%= maddwd(sp, @bitCast(bc4(@ptrCast(&acts[t][b].bsums[2 * p])))) << @splat(5);
        }
        var n: usize = 0;
        while (n < n_half) : (n += 1) {
            inline for (0..2) |kh| {
                // Scale index for element 128n + 32m + 4k is 8n + 2m + k/4.
                var sc: [4]V16i16 = undefined;
                inline for (0..4) |m| sc[m] = dupI8(blk + 32 + (8 * n + 2 * m + kh) * 8);
                // Two groups share a scale; 2 x 16002 still fits i16, so sum
                // their vpmaddubsw results before one vpmaddwd.
                var kp: usize = 0;
                while (kp < n_pairs) : (kp += 1) {
                    const k = kh * 4 + 2 * kp;
                    const b0 = blk + 160 + (n * 8 + k) * 96;
                    const la0 = load32(b0);
                    const lb0 = load32(b0 + 32);
                    const h0 = load32(b0 + 64);
                    const la1 = load32(b0 + 96);
                    const lb1 = load32(b0 + 128);
                    const h1 = load32(b0 + 160);
                    inline for (0..4) |m| {
                        const q0 = q6(m, la0, lb0, h0);
                        const q1 = q6(m, la1, lb1, h1);
                        inline for (0..NT) |t| {
                            const a = acts[t][b].qs[128 * n + 32 * m + 4 * k ..];
                            acc[t] += maddwd(maddubs(q0, bc4(a.ptr)) +% maddubs(q1, bc4(a.ptr + 4)), sc[m]);
                        }
                    }
                }
            }
        }
        const d8 = halfs(blk);
        inline for (0..NT) |t| out[t] = @mulAdd(V8f, d8 * @as(V8f, @splat(acts[t][b].d)), toF(acc[t]), out[t]);
    }
    return out;
}

/// Q4_0, Q4_1, Q5_0, Q5_1: 32-value blocks, one scale per row, plus a min
/// (Q4_1, Q5_1) or a zero point of 8 / 16 folded into the scale.
fn gemvLegacy(comptime t: GgmlType, w: [*]const u8, act: [*]const Q8B, nb: usize) V8f {
    @setEvalBranchQuota(100_000);
    const has_m = t == .q4_1 or t == .q5_1;
    const has_h = t == .q5_0 or t == .q5_1;
    const BS = comptime blockBytes(t);
    const qh_off = if (has_m) 32 else 16;
    const qs_off = qh_off + if (has_h) 32 else 0;
    const sixteen: V32u8 = @splat(0x10);
    const zero: V32u8 = @splat(0);
    var out: V8f = @splat(0);
    for (0..nb) |b| {
        const blk = w + b * BS;
        prefetchAhead(blk, BS);
        const y = &act[b];
        var s: V16i16 = @splat(0);
        var acc: V8i32 = @splat(0);
        inline for (0..4) |k| {
            const wv = load32(blk + qs_off + k * 32);
            var lo = wv & m4;
            var hi = wv >> @splat(4);
            if (has_h) {
                const hb = spreadRows(blk + qh_off + k * 8);
                lo |= @select(u8, (hb & h_lo) == h_lo, sixteen, zero);
                hi |= @select(u8, (hb & h_hi) == h_hi, sixteen, zero);
            }
            const p = maddubs(lo, bc4(y.qs[4 * k ..].ptr)) +% maddubs(hi, bc4(y.qs[16 + 4 * k ..].ptr));
            if (has_h) acc += maddwd(p, ones16) else s +%= p;
        }
        const d8 = halfs(blk);
        const c8: V8f = if (has_m) halfs(blk + 16) else d8 * @as(V8f, @splat(if (has_h) -16 else -8));
        const ints = if (has_h) acc else maddwd(s, ones16);
        out = @mulAdd(V8f, d8 * @as(V8f, @splat(y.d)), toF(ints), out);
        out = @mulAdd(V8f, c8, @splat(y.s), out);
    }
    return out;
}

fn gemmLegacy(comptime t: GgmlType, comptime NT: usize, w: [*]const u8, acts: [NT][*]const Q8B, nb: usize) [NT]V8f {
    @setEvalBranchQuota(100_000);
    const has_m = t == .q4_1 or t == .q5_1;
    const has_h = t == .q5_0 or t == .q5_1;
    const BS = comptime blockBytes(t);
    const qh_off = if (has_m) 32 else 16;
    const qs_off = qh_off + if (has_h) 32 else 0;
    const sixteen: V32u8 = @splat(0x10);
    const zero: V32u8 = @splat(0);
    var out: [NT]V8f = @splat(@splat(0));
    const n_k = hidden(4);
    for (0..nb) |b| {
        const blk = w + b * BS;
        prefetchAhead(blk, BS);
        // 4-bit values: the block's eight vpmaddubsw pair sums stay below
        // 8 * 3810 in i16. 5-bit ones reach 2 * 7874 per step: widen each step.
        var s: [NT]V16i16 = @splat(@splat(0));
        var acc: [NT]V8i32 = @splat(@splat(0));
        var k: usize = 0;
        while (k < n_k) : (k += 1) {
            const wv = load32(blk + qs_off + k * 32);
            var lo = wv & m4;
            var hi = wv >> @splat(4);
            if (has_h) {
                const hb = spreadRows(blk + qh_off + k * 8);
                lo |= @select(u8, (hb & h_lo) == h_lo, sixteen, zero);
                hi |= @select(u8, (hb & h_hi) == h_hi, sixteen, zero);
            }
            inline for (0..NT) |tk| {
                const y = &acts[tk][b];
                const p = maddubs(lo, bc4(y.qs[4 * k ..].ptr)) +% maddubs(hi, bc4(y.qs[16 + 4 * k ..].ptr));
                if (has_h) acc[tk] += maddwd(p, ones16) else s[tk] +%= p;
            }
        }
        const d8 = halfs(blk);
        // Offset per unit of activation sum: the min, or the zero point times d.
        const c8: V8f = if (has_m) halfs(blk + 16) else d8 * @as(V8f, @splat(if (has_h) -16 else -8));
        inline for (0..NT) |tk| {
            const y = &acts[tk][b];
            const ints = if (has_h) acc[tk] else maddwd(s[tk], ones16);
            out[tk] = @mulAdd(V8f, d8 * @as(V8f, @splat(y.d)), toF(ints), out[tk]);
            out[tk] = @mulAdd(V8f, c8, @splat(y.s), out[tk]);
        }
    }
    return out;
}

fn gemvQ8_0(w: [*]const u8, act: [*]const Q8B, nb: usize) V8f {
    @setEvalBranchQuota(100_000);
    var out: V8f = @splat(0);
    for (0..nb) |b| {
        const blk = w + b * 272;
        prefetchAhead(blk, 272);
        const y = &act[b];
        var acc: V8i32 = @splat(0);
        inline for (0..8) |k| {
            const wv: V32i8 = @bitCast(load32(blk + 16 + k * 32));
            const aw: V32u8 = @abs(wv);
            const sa = applySign(bc4(y.qs[4 * k ..].ptr), wv);
            acc += maddwd(maddubs(aw, sa), ones16);
        }
        const d8 = halfs(blk);
        out = @mulAdd(V8f, d8 * @as(V8f, @splat(y.d)), toF(acc), out);
    }
    return out;
}

fn gemmQ8_0(comptime NT: usize, w: [*]const u8, acts: [NT][*]const Q8B, nb: usize) [NT]V8f {
    @setEvalBranchQuota(100_000);
    var out: [NT]V8f = @splat(@splat(0));
    const n_k = hidden(8);
    for (0..nb) |b| {
        const blk = w + b * 272;
        prefetchAhead(blk, 272);
        var acc: [NT]V8i32 = @splat(@splat(0));
        var k: usize = 0;
        while (k < n_k) : (k += 1) {
            const wv: V32i8 = @bitCast(load32(blk + 16 + k * 32));
            const aw: V32u8 = @abs(wv);
            inline for (0..NT) |t| {
                const sa = applySign(bc4(acts[t][b].qs[4 * k ..].ptr), wv);
                acc[t] += maddwd(maddubs(aw, sa), ones16);
            }
        }
        const d8 = halfs(blk);
        inline for (0..NT) |t| out[t] = @mulAdd(V8f, d8 * @as(V8f, @splat(acts[t][b].d)), toF(acc[t]), out[t]);
    }
    return out;
}

inline fn castActs(comptime T: type, comptime NT: usize, acts: [NT][*]const u8) [NT][*]const T {
    var r: [NT][*]const T = undefined;
    inline for (0..NT) |k| r[k] = @ptrCast(@alignCast(acts[k]));
    return r;
}

/// 8 rows (one repacked group of `cols` elements) x NT activation rows.
pub inline fn gemm(comptime t: GgmlType, comptime NT: usize, group: [*]const u8, acts: [NT][*]const u8, cols: usize) [NT]V8f {
    if (NT == 1) {
        const v = switch (t) {
            .q4_k => gemvK45(false, group, @ptrCast(@alignCast(acts[0])), cols / 256),
            .q5_k => gemvK45(true, group, @ptrCast(@alignCast(acts[0])), cols / 256),
            .q6_k => gemvQ6K(group, @ptrCast(@alignCast(acts[0])), cols / 256),
            .q4_0, .q4_1, .q5_0, .q5_1 => gemvLegacy(t, group, @ptrCast(@alignCast(acts[0])), cols / 32),
            .q8_0 => gemvQ8_0(group, @ptrCast(@alignCast(acts[0])), cols / 32),
            else => unreachable,
        };
        return .{v};
    }
    return switch (t) {
        .q4_k => gemmK45(false, NT, group, castActs(Q8K, NT, acts), cols / 256),
        .q5_k => gemmK45(true, NT, group, castActs(Q8K, NT, acts), cols / 256),
        .q6_k => gemmQ6K(NT, group, castActs(Q8K, NT, acts), cols / 256),
        .q4_0, .q4_1, .q5_0, .q5_1 => gemmLegacy(t, NT, group, castActs(Q8B, NT, acts), cols / 32),
        .q8_0 => gemmQ8_0(NT, group, castActs(Q8B, NT, acts), cols / 32),
        else => unreachable,
    };
}

// ---------------------------------------------------------------------------
// 2-bit coarse copy of lm_head for vocab pruning
//
// Every weight becomes c * beta with c in {-3,-1,1,3}: Lloyd-Max thresholds
// for a Gaussian (0 and +-0.98 sigma) and a least-squares beta per row. Scoring
// all dimensions this coarsely ranks the true argmax far better than an exact
// dot over a quarter of them: over 3500 teacher-forced positions on
// Llama-3.2-1B and Qwen3-4B the winner's group never ranked below 249 (the
// fine pass takes 512), while the first-25%-of-dims score missed the argmax
// at 4-14% of positions.
//
// Per 8-row group and 32 dims: two 32-byte vectors, lane r holding row r.
// Byte j of a lane packs dims 4k+j (k = 0..3, 2 bits at shift 2k) of one
// 16-dim half, so shifting by 2k lines four consecutive dims up with a
// broadcast 4-byte activation slice. The 8 row scales (f32) follow the blocks.

/// Bytes of one 8-row group of `cols` elements in the coarse layout.
pub fn coarseGroupBytes(cols: usize) usize {
    return cols / 32 * 64 + 32;
}

/// Quantize 8 rows of equal length (a multiple of 32) into the coarse layout.
pub fn packCoarse(rows: [8][]const f32, dst: [*]u8) void {
    const cols = rows[0].len;
    const codes = dst[0 .. cols / 32 * 64];
    @memset(codes, 0);
    var scales: [8]f32 = undefined;
    for (rows, 0..) |row, r| {
        var ss: f64 = 0;
        for (row) |w| ss += @as(f64, w) * w;
        const t: f32 = @floatCast(0.9816 * @sqrt(ss / @as(f64, @floatFromInt(cols))));
        var wc: f64 = 0;
        var cc: f64 = 0;
        for (row, 0..) |w, d| {
            // Stored as u = (c + 3) / 2.
            const u: u8 = if (w < -t) 0 else if (w < 0) 1 else if (w < t) 2 else 3;
            const c: f64 = @floatFromInt(@as(i32, u) * 2 - 3);
            wc += c * w;
            cc += c * c;
            const e = d % 16;
            codes[d / 32 * 64 + (d % 32) / 16 * 32 + r * 4 + e % 4] |= u << @as(u3, @intCast(2 * (e / 4)));
        }
        scales[r] = @floatCast(wc / cc);
    }
    @memcpy(dst[codes.len..][0..32], std.mem.asBytes(&scales));
}

/// Coarse scores beta_r * sum_d c_rd * x_d of one group against a Q8B row;
/// `sum_s` is the sum of the row's per-block `s`.
pub fn coarseScores(w: [*]const u8, act: [*]const Q8B, nb: usize, sum_s: f32) V8f {
    const m2: V32u8 = @splat(3);
    var acc: V8f = @splat(0);
    for (0..nb) |b| {
        const blk = w + b * 64;
        prefetchAhead(blk, 64);
        const y = &act[b];
        // u <= 3: eight vpmaddubsw pair sums stay below 8 * 762 in i16.
        var s: V16i16 = @splat(0);
        inline for (0..2) |h| {
            const v = load32(blk + h * 32);
            inline for (0..4) |k| {
                const u = (v >> @splat(2 * k)) & m2;
                s +%= maddubs(u, bc4(y.qs[16 * h + 4 * k ..].ptr));
            }
        }
        acc = @mulAdd(V8f, @splat(y.d), toF(maddwd(s, ones16)), acc);
    }
    // sum c*x = 2 * sum u*x - 3 * sum x.
    const beta = loadVec(V8f, w + nb * 64);
    return beta * (acc * @as(V8f, @splat(2)) - @as(V8f, @splat(3 * sum_s)));
}

// ---------------------------------------------------------------------------

fn checkType(comptime t: GgmlType) !void {
    const cols = 512;
    const l = t.layout().?;
    const rb = cols / l.block * l.bytes;
    var prng = std.Random.DefaultPrng.init(@backingInt(t) + 17);
    const r = prng.random();
    var src: [8 * (512 / 32) * 34]u8 align(32) = undefined; // >= 8 * rb for every type
    r.bytes(src[0 .. 8 * rb]);
    for (0..8 * cols / l.block) |b| {
        const blk = src[b * l.bytes ..][0..l.bytes];
        const put = struct {
            fn f(p: []u8, v: f32) void {
                std.mem.writeInt(u16, p[0..2], @bitCast(@as(f16, @floatCast(v))), .little);
            }
        }.f;
        switch (t) {
            .q4_0, .q5_0, .q8_0 => put(blk[0..2], 0.01 + r.float(f32) * 0.05),
            .q4_1, .q5_1 => {
                put(blk[0..2], 0.01 + r.float(f32) * 0.05);
                put(blk[2..4], 0.01 + r.float(f32) * 0.05);
            },
            .q4_k, .q5_k => {
                put(blk[0..2], 0.01 + r.float(f32) * 0.05);
                put(blk[2..4], 0.01 + r.float(f32) * 0.05);
            },
            .q6_k => put(blk[208..210], 0.01 + r.float(f32) * 0.05),
            else => unreachable,
        }
    }
    var rows: [8][*]const u8 = undefined;
    for (&rows, 0..) |*p, i| p.* = src[i * rb ..].ptr;
    var dst: [8 * (512 / 256) * 1696]u8 align(32) = undefined;
    repackGroup(t, rows, cols / l.block, &dst);

    const NT = 3;
    const kind = quant.actKind(t);
    var ak: [NT][cols / 256]Q8K = undefined;
    var ab: [NT][cols / 32]Q8B = undefined;
    var acts: [NT][*]const u8 = undefined;
    for (0..NT) |k| {
        var x: [cols]f32 = undefined;
        for (&x) |*v| v.* = r.floatNorm(f32);
        quant.quantizeQ8K(&x, &ak[k]);
        quant.quantizeQ8B(&x, &ab[k]);
        acts[k] = if (kind == .q8k) @ptrCast(&ak[k]) else @ptrCast(&ab[k]);
    }
    const got = gemm(t, NT, &dst, acts, cols);
    const got1 = gemm(t, 1, &dst, acts[1..2].*, cols);
    for (0..NT) |k| {
        const g: [8]f32 = got[k];
        const g1: [8]f32 = got1[0];
        for (0..8) |row| {
            const want = quant.dot(t, rows[row], acts[k], cols);
            const tol = 1e-3 * @max(1.0, @abs(want));
            if (@abs(g[row] - want) > tol or (k == 1 and @abs(g1[row] - want) > tol)) {
                std.debug.print("{s} tok {d} row {d}: got {d} want {d}\n", .{ @tagName(t), k, row, g[row], want });
                return error.KernelMismatch;
            }
        }
    }
}

test "x8 kernels match per-row kernels" {
    inline for (.{ GgmlType.q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q4_k, .q5_k, .q6_k }) |t| try checkType(t);
}

test "2-bit coarse scores match the decoded weights" {
    const cols = 256;
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    var rows_f: [8][cols]f32 = undefined;
    var rows: [8][]const f32 = undefined;
    for (&rows_f, 0..) |*row, i| {
        const scale = 0.02 + 0.01 * @as(f32, @floatFromInt(i));
        for (row) |*v| v.* = r.floatNorm(f32) * scale;
        rows[i] = row;
    }
    var buf: [coarseGroupBytes(cols)]u8 align(32) = undefined;
    packCoarse(rows, &buf);

    var x: [cols]f32 = undefined;
    for (&x) |*v| v.* = r.floatNorm(f32);
    var act: [cols / 32]Q8B = undefined;
    quant.quantizeQ8B(&x, &act);
    var sum_s: f32 = 0;
    for (act) |a| sum_s += a.s;
    const got: [8]f32 = coarseScores(&buf, &act, cols / 32, sum_s);

    // Independent decode of the documented layout.
    const beta: [8]f32 = @bitCast(buf[cols / 32 * 64 ..][0..32].*);
    for (0..8) |row| {
        var want: f64 = 0;
        var err2: f64 = 0;
        var norm2: f64 = 0;
        for (0..cols) |d| {
            const e = d % 16;
            const byte = buf[d / 32 * 64 + (d % 32) / 16 * 32 + row * 4 + e % 4];
            const u = (byte >> @as(u3, @intCast(2 * (e / 4)))) & 3;
            const wq = beta[row] * @as(f32, @floatFromInt(@as(i32, u) * 2 - 3));
            const blk = act[d / 32];
            want += wq * blk.d * @as(f32, @floatFromInt(blk.qs[d % 32]));
            const diff = wq - rows_f[row][d];
            err2 += diff * diff;
            norm2 += rows_f[row][d] * rows_f[row][d];
        }
        try std.testing.expectApproxEqAbs(want, got[row], 1e-3 * @max(1.0, @abs(want)));
        // 2-bit Lloyd-Max keeps a Gaussian row within ~0.34 relative L2 error.
        try std.testing.expect(@sqrt(err2 / norm2) < 0.4);
    }
}
