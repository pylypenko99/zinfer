// rope.zig - Rotary Position Embedding (RoPE) and NTK-Aware Frequency Scaling
// Zero-dependency pure Zig module.

const std = @import("std");

/// Computes inverse frequencies for RoPE: \theta_i = base^(-2i/d) * scale.
/// If optional frequency factors (e.g. from rope_freqs.weight) are provided, divides by them.
pub fn ropeFreqs(base: f32, scale: f32, factors: ?[]const f32, out: []f32) void {
    const n_rot = out.len * 2;
    for (out, 0..) |*v, i| {
        const e = -2.0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n_rot));
        var w = std.math.pow(f64, base, e) * scale;
        if (factors) |fr| w /= fr[i];
        v.* = @floatCast(w);
    }
}

/// Dynamically scales RoPE base frequency according to NTK-aware context extension:
/// \theta_{ntk} = \theta_{base} * \alpha^{d / (d - 2)}, where \alpha = ctx / ctx_train.
/// Prevents perplexity explosion and attention score degradation when context length
/// exceeds the original training sequence length.
pub fn ntkAwareBase(rope_base: f32, ctx: usize, ctx_train: usize, head_dim: usize) f32 {
    if (ctx > ctx_train and ctx_train > 0 and head_dim > 2) {
        const alpha = @as(f64, @floatFromInt(ctx)) / @as(f64, @floatFromInt(ctx_train));
        const exponent = @as(f64, @floatFromInt(head_dim)) / @as(f64, @floatFromInt(head_dim - 2));
        const scale = std.math.pow(f64, alpha, exponent);
        return @floatCast(@as(f64, rope_base) * scale);
    }
    return rope_base;
}

/// Precomputes cosine and sine tables for `ntok` tokens starting at position `pos0`.
/// Table layout per token: [half cos values, half sin values] where half = n_rot / 2.
pub fn precomputeRopeTable(table: []f32, freqs: []const f32, pos0: usize, ntok: usize, n_rot: usize) void {
    const half = n_rot / 2;
    for (0..ntok) |t| {
        const p: f32 = @floatFromInt(pos0 + t);
        const cs = table[t * n_rot ..];
        for (freqs, 0..) |f, i| {
            const a = p * f;
            cs[i] = @cos(a);
            cs[half + i] = @sin(a);
        }
    }
}

/// Applies 2D rotary embedding in-place to vector `v` of length (n_heads * hd).
/// Supports both NeoX style (split halves: [0..half] and [half..n_rot]) and
/// standard interleaved style (pairs: 2*i and 2*i+1).
pub fn applyRope(v: []f32, cs: []const f32, hd: usize, n_rot: usize, rope_neox: bool) void {
    const half = n_rot / 2;
    var h: usize = 0;
    while (h < v.len) : (h += hd) {
        const x = v[h..][0..hd];
        for (0..half) |i| {
            const ia = if (rope_neox) i else 2 * i;
            const ib = if (rope_neox) i + half else 2 * i + 1;
            const a = x[ia];
            const b = x[ib];
            x[ia] = a * cs[i] - b * cs[half + i];
            x[ib] = a * cs[half + i] + b * cs[i];
        }
    }
}

// --------------------------------------------------------------------------
// Tests

test "RoPE mathematical reference and NTK-aware frequency scaling" {
    const head_dim = 128;
    const half = head_dim / 2;
    const base: f32 = 10000.0;

    var freqs: [half]f32 = undefined;
    ropeFreqs(base, 1.0, null, &freqs);

    // Verify monotonic frequency decrease
    try std.testing.expect(freqs[0] > freqs[half - 1]);
    try std.testing.expect(freqs[0] >= 0.99 and freqs[0] <= 1.01);

    // Verify NTK scaling preserves low frequencies while scaling base
    const ctx_train = 4096;
    const ctx_extended = 16384;
    const base_ntk = ntkAwareBase(base, ctx_extended, ctx_train, head_dim);
    try std.testing.expect(base_ntk > base);

    var freqs_ntk: [half]f32 = undefined;
    ropeFreqs(base_ntk, 1.0, null, &freqs_ntk);
    // Wavelength at lowest frequency is scaled
    try std.testing.expect(freqs_ntk[half - 1] < freqs[half - 1]);

    // Test RoPE orthogonal rotation preserves vector norm
    var table: [head_dim]f32 = undefined;
    precomputeRopeTable(&table, &freqs, 10, 1, head_dim);

    var v: [head_dim]f32 = undefined;
    for (&v, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i + 1)));

    var norm_before: f32 = 0;
    for (v) |x| norm_before += x * x;

    applyRope(&v, &table, head_dim, head_dim, true);

    var norm_after: f32 = 0;
    for (v) |x| norm_after += x * x;

    try std.testing.expectApproxEqRel(norm_before, norm_after, 1e-4);
}
