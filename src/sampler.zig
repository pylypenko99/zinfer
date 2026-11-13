const std = @import("std");

pub const Params = struct {
    temp: f32 = 0.8,
    top_k: usize = 40,
    top_p: f32 = 0.95,
    min_p: f32 = 0.05,
    seed: u64 = 0,
};

const Cand = struct { id: u32, v: f32 };

pub const Sampler = struct {
    p: Params,
    rng: std.Random.DefaultPrng,
    cand: []Cand,

    pub fn init(gpa: std.mem.Allocator, vocab: usize, p: Params) !Sampler {
        return .{ .p = p, .rng = .init(p.seed), .cand = try gpa.alloc(Cand, vocab) };
    }

    pub fn deinit(s: *Sampler, gpa: std.mem.Allocator) void {
        gpa.free(s.cand);
    }

    pub fn sample(s: *Sampler, logits: []const f32) u32 {
        if (s.p.temp <= 0) return argmax(logits);

        // Top-k via a small sorted buffer (descending).
        var n: usize = 0;
        const k = if (s.p.top_k == 0 or s.p.top_k > logits.len) logits.len else s.p.top_k;
        if (k == logits.len) {
            for (logits, 0..) |v, i| s.cand[i] = .{ .id = @intCast(i), .v = v };
            n = logits.len;
            std.mem.sort(Cand, s.cand[0..n], {}, gt);
        } else {
            for (logits, 0..) |v, i| {
                if (n == k and v <= s.cand[k - 1].v) continue;
                var j = if (n < k) n else k - 1;
                if (n < k) n += 1;
                while (j > 0 and s.cand[j - 1].v < v) : (j -= 1) s.cand[j] = s.cand[j - 1];
                s.cand[j] = .{ .id = @intCast(i), .v = v };
            }
        }
        const c = s.cand[0..n];

        const inv_t = 1.0 / s.p.temp;
        const mx = c[0].v;
        var sum: f32 = 0;
        for (c) |*e| {
            e.v = @exp((e.v - mx) * inv_t);
            sum += e.v;
        }
        // min-p relative to the top probability (which is exp(0) = 1 before normalising).
        var keep = n;
        if (s.p.min_p > 0) {
            for (c, 0..) |e, i| if (e.v < s.p.min_p) {
                keep = @max(i, 1);
                break;
            };
        }
        // top-p
        var total: f32 = 0;
        for (c[0..keep]) |e| total += e.v;
        if (s.p.top_p < 1) {
            var acc: f32 = 0;
            for (c[0..keep], 0..) |e, i| {
                acc += e.v;
                if (acc >= s.p.top_p * total) {
                    keep = i + 1;
                    break;
                }
            }
            total = 0;
            for (c[0..keep]) |e| total += e.v;
        }
        var r = s.rng.random().float(f32) * total;
        for (c[0..keep]) |e| {
            r -= e.v;
            if (r <= 0) return e.id;
        }
        return c[keep - 1].id;
    }

    fn gt(_: void, a: Cand, b: Cand) bool {
        return a.v > b.v;
    }
};

pub fn argmax(x: []const f32) u32 {
    var best: usize = 0;
    for (x, 0..) |v, i| if (v > x[best]) {
        best = i;
    };
    return @intCast(best);
}
