const std = @import("std");

pub const Params = struct {
    temp: f32 = 0.8,
    top_k: usize = 40,
    top_p: f32 = 0.95,
    min_p: f32 = 0.05,
    /// Divides positive / multiplies negative logits of recent tokens (1 = off).
    repeat_penalty: f32 = 1.0,
    seed: u64 = 0,
};

/// Tokens considered by the repetition penalty.
pub const repeat_last_n = 64;

const Cand = struct { id: u32, v: f32 };

/// Filter chain in llama.cpp's default order: repetition penalty, top-k,
/// top-p, min-p (on untempered probabilities), then temperature.
pub const Sampler = struct {
    p: Params,
    rng: std.Random.DefaultPrng,
    cand: []Cand,
    recent: [repeat_last_n]u32 = undefined,
    n_recent: usize = 0,

    pub fn init(gpa: std.mem.Allocator, vocab: usize, p: Params) !Sampler {
        return .{ .p = p, .rng = .init(p.seed), .cand = try gpa.alloc(Cand, vocab) };
    }

    pub fn deinit(s: *Sampler, gpa: std.mem.Allocator) void {
        gpa.free(s.cand);
    }

    /// Record a token that entered the context (prompt or generated).
    pub fn accept(s: *Sampler, tok: u32) void {
        s.recent[s.n_recent % repeat_last_n] = tok;
        s.n_recent += 1;
    }

    fn applyPenalty(s: *const Sampler, logits: []f32) void {
        if (s.p.repeat_penalty == 1) return;
        const n = @min(s.n_recent, repeat_last_n);
        for (s.recent[0..n], 0..) |tok, i| {
            // Each distinct token is penalized once.
            if (std.mem.indexOfScalar(u32, s.recent[0..i], tok) != null) continue;
            const l = &logits[tok];
            l.* = if (l.* > 0) l.* / s.p.repeat_penalty else l.* * s.p.repeat_penalty;
        }
    }

    /// Pick the next token; `logits` is modified in place by the penalty.
    pub fn sample(s: *Sampler, logits: []f32) u32 {
        s.applyPenalty(logits);
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
        const mx = c[0].v;

        // top-p, then min-p, on untempered probabilities (relative to the top one).
        var keep = n;
        if (s.p.top_p < 1) {
            var total: f32 = 0;
            for (c) |e| total += @exp(e.v - mx);
            var acc: f32 = 0;
            for (c, 0..) |e, i| {
                acc += @exp(e.v - mx);
                if (acc >= s.p.top_p * total) {
                    keep = i + 1;
                    break;
                }
            }
        }
        if (s.p.min_p > 0) {
            for (c[0..keep], 0..) |e, i| if (@exp(e.v - mx) < s.p.min_p) {
                keep = @max(i, 1);
                break;
            };
        }

        // Temperature on the survivors, then draw.
        const inv_t = 1.0 / s.p.temp;
        var total: f32 = 0;
        for (c[0..keep]) |*e| {
            e.v = @exp((e.v - mx) * inv_t);
            total += e.v;
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
