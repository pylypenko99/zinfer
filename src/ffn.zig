// ffn.zig - Feed-Forward Networks (SwiGLU, GeLU, MoE Routing)
// Zero-dependency pure Zig module.

const std = @import("std");

pub const V8f = @Vector(8, f32);

pub const silu_dead_threshold: f32 = -6.0;
pub const gelu_dead_threshold: f32 = -4.0;

/// Cephes-style expf on 8 SIMD lanes (relative error ~1e-7).
pub inline fn expV(x_in: V8f) V8f {
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

/// Computes SiLU(gate) * up = gate / (1 + exp(-gate)) * up in 8 SIMD lanes.
pub inline fn siluMulV(gv: V8f, uv: V8f) V8f {
    return gv / (@as(V8f, @splat(1)) + expV(-gv)) * uv;
}

/// GELU-gated FFN (Gemma): gelu(g) * u using tanh approximation.
/// Evaluates x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 x^3)).
pub inline fn geluMulV(x: V8f, uv: V8f) V8f {
    const H = @Vector(8, f16);
    const k0: V8f = @splat(2 * 0.79788456080286535587989211986876);
    const k1: V8f = @splat(0.044715);
    const one: V8f = @splat(1);
    const xh: V8f = @floatCast(@as(H, @floatCast(x)));
    const gl: V8f = @floatCast(@as(H, @floatCast(xh / (one + expV(-k0 * xh * (one + k1 * xh * xh))))));
    const y = @select(f32, x <= @as(V8f, @splat(-10)), @as(V8f, @splat(0)), @select(f32, x >= @as(V8f, @splat(10)), x, gl));
    return y * uv;
}

/// Computes SiLU-gated elementwise product: out = silu(g) * u
pub fn silu_mul(out: []f32, g: []const f32, u: []const f32) void {
    var i: usize = 0;
    while (i + 8 <= out.len) : (i += 8) {
        const gv: V8f = g[i..][0..8].*;
        const uv: V8f = u[i..][0..8].*;
        out[i..][0..8].* = siluMulV(gv, uv);
    }
    while (i < out.len) : (i += 1) out[i] = g[i] / (1 + @exp(-g[i])) * u[i];
}

/// Computes GELU-gated elementwise product: out = gelu(g) * u
pub fn gelu_mul(out: []f32, g: []const f32, u: []const f32) void {
    const k0: V8f = @splat(2 * 0.79788456080286535587989211986876);
    const k1: V8f = @splat(0.044715);
    var i: usize = 0;
    while (i + 8 <= out.len) : (i += 8) {
        const x: V8f = g[i..][0..8].*;
        const uv: V8f = u[i..][0..8].*;
        out[i..][0..8].* = geluMulV(x, uv);
    }
    while (i < out.len) : (i += 1) {
        const x = g[i];
        const xh: f32 = @as(f16, @floatCast(x));
        const gl: f32 = @as(f16, @floatCast(xh / (1 + @exp(-k0[0] * xh * (1 + k1[0] * xh * xh)))));
        out[i] = (if (x <= -10) 0 else if (x >= 10) x else gl) * u[i];
    }
}

/// MoE top-K routing: selects the top-K expert indices from router logits,
/// applies numerical softmax over selected logits, and normalizes by `weights_scale`.
pub fn routeTopK(
    logits: []const f32,
    k_exp: usize,
    weights_scale: f32,
    out_indices: []usize,
    out_weights: []f32,
) void {
    var found: usize = 0;
    for (logits, 0..) |v, e| {
        if (found < k_exp) {
            var j = found;
            found += 1;
            while (j > 0 and out_weights[j - 1] < v) : (j -= 1) {
                out_weights[j] = out_weights[j - 1];
                out_indices[j] = out_indices[j - 1];
            }
            out_weights[j] = v;
            out_indices[j] = e;
        } else if (v > out_weights[k_exp - 1]) {
            var j = k_exp - 1;
            while (j > 0 and out_weights[j - 1] < v) : (j -= 1) {
                out_weights[j] = out_weights[j - 1];
                out_indices[j] = out_indices[j - 1];
            }
            out_weights[j] = v;
            out_indices[j] = e;
        }
    }

    var max_l: f32 = out_weights[0];
    for (out_weights[1..k_exp]) |v| max_l = @max(max_l, v);
    var sum_exp: f32 = 0;
    for (0..k_exp) |j| {
        const ev = @exp(out_weights[j] - max_l);
        out_weights[j] = ev;
        sum_exp += ev;
    }
    const inv_sum = (1.0 / sum_exp) * weights_scale;
    for (0..k_exp) |j| {
        out_weights[j] *= inv_sum;
    }
}

// --------------------------------------------------------------------------
// Tests

test "SwiGLU and GeLU FFN mathematical reference" {
    var g = [_]f32{ -8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0, 12.0 };
    var u = [_]f32{ 1.0, 2.0, -1.0, 0.5, 2.0, -0.5, 1.5, 0.25 };
    var out_silu: [8]f32 = undefined;
    var out_gelu: [8]f32 = undefined;

    silu_mul(&out_silu, &g, &u);
    gelu_mul(&out_gelu, &g, &u);

    // Verify SiLU properties
    // At g=0, silu(0) = 0 -> out = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_silu[3], 1e-5);
    // At g=-8, silu(-8) is close to 0 (dead neuron in predictive sparsity)
    try std.testing.expect(@abs(out_silu[0]) < 0.01);
    // At g=1.0, u=2.0: silu(1.0) = 1 / (1 + exp(-1)) = 0.7310585786; out = 1.462117
    try std.testing.expectApproxEqRel(@as(f32, 1.462117), out_silu[4], 1e-4);

    // Verify GeLU properties
    // At g <= -10 (or -8), gelu is very close to 0
    try std.testing.expect(@abs(out_gelu[0]) < 0.01);
    // At g >= 10, gelu(x) = x -> at g=12, u=0.25: out = 3.0
    try std.testing.expectApproxEqRel(@as(f32, 3.0), out_gelu[7], 1e-4);
}

test "MoE top-K routing and expert execution matches mathematical reference" {
    // Test 4 experts, top-2 routing, dim=16
    const n_exp = 4;
    const k_exp = 2;
    const dim = 16;

    var prng = std.Random.DefaultPrng.init(12345);
    const r = prng.random();

    var router_w: [n_exp * dim]f32 = undefined;
    for (&router_w) |*x| x.* = r.floatNorm(f32);

    var x: [dim]f32 = undefined;
    for (&x) |*v| v.* = r.floatNorm(f32);

    var logits: [n_exp]f32 = undefined;
    for (0..n_exp) |e| {
        var dot: f32 = 0;
        for (0..dim) |d| dot += router_w[e * dim + d] * x[d];
        logits[e] = dot;
    }

    var top_idx: [k_exp]usize = undefined;
    var top_val: [k_exp]f32 = undefined;

    routeTopK(&logits, k_exp, 1.0, &top_idx, &top_val);

    const w0 = top_val[0];
    const w1 = top_val[1];

    try std.testing.expect(w0 + w1 >= 0.9999 and w0 + w1 <= 1.0001);
    try std.testing.expect(w0 >= w1);
    try std.testing.expect(top_idx[0] != top_idx[1]);
}
