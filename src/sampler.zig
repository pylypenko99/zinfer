const std = @import("std");

pub const Params = struct {
    temp: f32 = 0.8,
    top_k: usize = 40,
    top_p: f32 = 0.95,
    min_p: f32 = 0.05,
    /// Divides positive / multiplies negative logits of recent tokens (1 = off).
    repeat_penalty: f32 = 1.0,
    seed: u64 = 0,

    /// DRY (Don't Repeat Yourself) repetition penalty multiplier (0 = off, 0.8 recommended).
    dry_multiplier: f32 = 0.0,
    /// DRY exponential penalty base.
    dry_base: f32 = 1.75,
    /// Minimum matching sequence length before DRY penalty applies.
    dry_allowed_length: usize = 2,
    /// Lookback context length in tokens for repetition detection.
    dry_range: usize = 1024,
};

/// Tokens considered by the repetition penalty.
pub const repeat_last_n = 64;
pub const max_history = 4096;
pub const max_breakers = 16;

const Cand = struct { id: u32, v: f32 };

/// Filter chain in llama.cpp's default order: repetition penalty, DRY penalty,
/// top-k, top-p, min-p (on untempered probabilities), then temperature.
pub const Sampler = struct {
    p: Params,
    rng: std.Random.DefaultPrng,
    cand: []Cand,
    recent: [repeat_last_n]u32 = undefined,
    n_recent: usize = 0,
    history: [max_history]u32 = undefined,
    n_history: usize = 0,
    breakers: [max_breakers]u32 = undefined,
    n_breakers: usize = 0,

    pub fn init(gpa: std.mem.Allocator, vocab: usize, p: Params) !Sampler {
        return .{
            .p = p,
            .rng = .init(p.seed),
            .cand = try gpa.alloc(Cand, vocab),
            .n_recent = 0,
            .n_history = 0,
            .n_breakers = 0,
        };
    }

    pub fn deinit(s: *Sampler, gpa: std.mem.Allocator) void {
        gpa.free(s.cand);
    }

    pub fn reset(s: *Sampler) void {
        s.n_recent = 0;
        s.n_history = 0;
    }

    pub fn addBreaker(s: *Sampler, tok: u32) void {
        if (s.n_breakers < max_breakers) {
            s.breakers[s.n_breakers] = tok;
            s.n_breakers += 1;
        }
    }

    pub fn isBreaker(s: *const Sampler, tok: u32) bool {
        for (s.breakers[0..s.n_breakers]) |b| {
            if (b == tok) return true;
        }
        return false;
    }

    /// Record a token that entered the context (prompt or generated).
    pub fn accept(s: *Sampler, tok: u32) void {
        s.recent[s.n_recent % repeat_last_n] = tok;
        s.n_recent += 1;
        s.history[s.n_history & (max_history - 1)] = tok;
        s.n_history += 1;
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

    /// DRY (Don't Repeat Yourself) Sampler: linear-time Z-algorithm based repetition
    /// suppression that penalizes continuation tokens extending matching previous n-grams.
    pub fn applyDry(s: *const Sampler, logits: []f32) void {
        if (s.p.dry_multiplier == 0.0 or s.p.dry_base < 1.0 or s.p.dry_range == 0) return;
        const L: usize = @min(s.n_history, @min(s.p.dry_range, max_history));
        if (L <= s.p.dry_allowed_length) return;

        const cur_pos = s.n_history - 1;

        // Step 1: Look for sequence breakers in reversed stream R to limit repetition depth
        var rep_limit: usize = L;
        for (0..L) |k| {
            const tok = s.history[(cur_pos -% k) & (max_history - 1)];
            if (s.isBreaker(tok)) {
                rep_limit = k;
                break;
            }
        }
        if (rep_limit < s.p.dry_allowed_length) return;

        // Step 2: Z-algorithm on the reversed token sequence.
        // R[k] is s.history[(cur_pos -% k) & (max_history - 1)].
        // z[i] is the length of common prefix between R[0..] and R[i..].
        // This runs in exact linear time O(L) without dynamic allocations.
        var z: [max_history]usize = undefined;
        z[0] = 0;
        var l: usize = 0;
        var r: usize = 0;
        for (1..L) |i| {
            if (i <= r) {
                z[i] = @min(r - i + 1, z[i - l]);
            } else {
                z[i] = 0;
            }
            const max_z = @min(L - i, rep_limit);
            while (z[i] < max_z) {
                const tok_pref = s.history[(cur_pos -% z[i]) & (max_history - 1)];
                const tok_curr = s.history[(cur_pos -% (i + z[i])) & (max_history - 1)];
                if (tok_pref != tok_curr) break;
                z[i] += 1;
            }
            if (i + z[i] > 0 and i + z[i] - 1 > r) {
                l = i;
                r = i + z[i] - 1;
            }
        }

        // Step 3: Collect maximum repeat length for each continuation token
        const max_penalized = 128;
        var pen_toks: [max_penalized]u32 = undefined;
        var pen_lens: [max_penalized]usize = undefined;
        var n_pen: usize = 0;

        for (1..L) |k| {
            const rlen = z[k];
            if (rlen >= s.p.dry_allowed_length) {
                const cont_tok = s.history[(cur_pos -% (k - 1)) & (max_history - 1)];
                if (s.isBreaker(cont_tok)) continue;

                var found = false;
                for (pen_toks[0..n_pen], 0..) |pt, pi| {
                    if (pt == cont_tok) {
                        pen_lens[pi] = @max(pen_lens[pi], rlen);
                        found = true;
                        break;
                    }
                }
                if (!found and n_pen < max_penalized) {
                    pen_toks[n_pen] = cont_tok;
                    pen_lens[n_pen] = rlen;
                    n_pen += 1;
                }
            }
        }

        // Step 4: Apply exponential penalty to logits
        const max_exp: f32 = if (s.p.dry_base > 1.0001) 88.0 / @log(s.p.dry_base) else 100.0;
        for (pen_toks[0..n_pen], pen_lens[0..n_pen]) |tok, rlen| {
            if (tok < logits.len) {
                const repeat_exp: f32 = @floatFromInt(rlen - s.p.dry_allowed_length);
                const exp = @min(repeat_exp, max_exp);
                const penalty = s.p.dry_multiplier * std.math.pow(f32, s.p.dry_base, exp);
                logits[tok] -= penalty;
            }
        }
    }

    /// Pick the next token with its confidence margin; `logits` is modified in place by the penalty.
    pub fn sampleWithMargin(s: *Sampler, logits: []f32) SampleResult {
        s.applyPenalty(logits);
        s.applyDry(logits);
        if (s.p.temp <= 0) return argmaxWithMargin(logits);

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
        const margin: f32 = if (n >= 2) mx - c[1].v else 10.0;

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
            if (r <= 0) return .{ .tok = e.id, .margin = margin };
        }
        return .{ .tok = c[keep - 1].id, .margin = margin };
    }

    /// Pick the next token; `logits` is modified in place by the penalty.
    pub fn sample(s: *Sampler, logits: []f32) u32 {
        return s.sampleWithMargin(logits).tok;
    }

    fn gt(_: void, a: Cand, b: Cand) bool {
        return a.v > b.v;
    }
};

/// Speculative Prompt Lookup Decoding (Tree of Hypotheses & n-gram candidate draft).
/// Searches the context history for matching n-grams, builds candidate hypothesis branches,
/// and ranks candidates based on n-gram order, occurrence frequency, and recency.
pub const max_tree_branches: usize = 4;
pub const max_branch_len: usize = 8;

pub const CandidateBranch = struct {
    tokens: [max_branch_len]u32,
    len: usize,
    score: u32,
    freq: u16,
};

pub const CandidateTree = struct {
    branches: [max_tree_branches]CandidateBranch,
    count: usize,

    pub fn init() CandidateTree {
        return .{
            .branches = undefined,
            .count = 0,
        };
    }

    pub fn best(self: *const CandidateTree) ?*const CandidateBranch {
        if (self.count == 0) return null;
        var best_idx: usize = 0;
        var best_score: u32 = self.branches[0].score;
        for (self.branches[1..self.count], 1..) |*b, i| {
            if (b.score > best_score) {
                best_score = b.score;
                best_idx = i;
            }
        }
        return &self.branches[best_idx];
    }

    pub fn secondBest(self: *const CandidateTree) ?*const CandidateBranch {
        if (self.count < 2) return null;
        var best_idx: usize = 0;
        var best_score: u32 = self.branches[0].score;
        for (self.branches[1..self.count], 1..) |*b, i| {
            if (b.score > best_score) {
                best_score = b.score;
                best_idx = i;
            }
        }
        var sec_idx: ?usize = null;
        var sec_score: u32 = 0;
        for (self.branches[0..self.count], 0..) |*b, i| {
            if (i == best_idx) continue;
            if (sec_idx == null or b.score > sec_score) {
                sec_score = b.score;
                sec_idx = i;
            }
        }
        return if (sec_idx) |idx| &self.branches[idx] else null;
    }

    pub fn findAlternative(self: *const CandidateTree, first_tok: u32) ?*const CandidateBranch {
        for (self.branches[0..self.count]) |*b| {
            if (b.len > 1 and b.tokens[0] == first_tok) return b;
        }
        return null;
    }
};

pub const PromptLookup = struct {
    fn addOrUpdateBranch(tree: *CandidateTree, tokens: []const u32, base_score: u32, recency: u32) void {
        if (tokens.len == 0) return;
        const len = @min(tokens.len, max_branch_len);
        for (tree.branches[0..tree.count]) |*b| {
            const common = @min(b.len, len);
            if (std.mem.eql(u32, b.tokens[0..common], tokens[0..common])) {
                b.freq += 1;
                b.score += 25;
                if (len > b.len) {
                    @memcpy(b.tokens[b.len..len], tokens[b.len..len]);
                    b.len = len;
                }
                return;
            }
        }
        if (tree.count < max_tree_branches) {
            var b = &tree.branches[tree.count];
            @memcpy(b.tokens[0..len], tokens[0..len]);
            b.len = len;
            b.freq = 1;
            b.score = base_score + recency;
            tree.count += 1;
        }
    }

    pub fn buildTree(history: []const u32, max_k: usize, tree: *CandidateTree) void {
        tree.count = 0;
        if (history.len < 2 or max_k == 0) return;

        // 1. Try 4-gram match across all historical occurrences
        if (history.len >= 8) {
            const target = history[history.len - 4 ..][0..4];
            const search_len = history.len - 4;
            var idx: usize = search_len - 4 + 1;
            var matches: usize = 0;
            while (idx > 0 and matches < 8) {
                idx -= 1;
                if (std.mem.eql(u32, history[idx .. idx + 4], target)) {
                    matches += 1;
                    const start = idx + 4;
                    const avail = @min(max_k, search_len - start);
                    if (avail > 0) {
                        const recency: u32 = @intCast(@min(20, (idx * 20) / (search_len + 1)));
                        addOrUpdateBranch(tree, history[start .. start + avail], 100, recency);
                    }
                }
            }
        }

        // 2. Try 3-gram match if fewer than 2 distinct candidate branches found
        if (tree.count < 2 and history.len >= 6) {
            const target = history[history.len - 3 ..][0..3];
            const search_len = history.len - 3;
            var idx: usize = search_len - 3 + 1;
            var matches: usize = 0;
            while (idx > 0 and matches < 8) {
                idx -= 1;
                if (std.mem.eql(u32, history[idx .. idx + 3], target)) {
                    matches += 1;
                    const start = idx + 3;
                    const avail = @min(max_k, search_len - start);
                    if (avail > 0) {
                        const recency: u32 = @intCast(@min(20, (idx * 20) / (search_len + 1)));
                        addOrUpdateBranch(tree, history[start .. start + avail], 60, recency);
                    }
                }
            }
        }

        // 3. Try 2-gram match (draft at most 2 tokens)
        if (tree.count < 2 and history.len >= 4) {
            const target = history[history.len - 2 ..][0..2];
            const search_len = history.len - 2;
            var idx: usize = search_len - 2 + 1;
            var matches: usize = 0;
            while (idx > 0 and matches < 8) {
                idx -= 1;
                if (std.mem.eql(u32, history[idx .. idx + 2], target)) {
                    matches += 1;
                    const start = idx + 2;
                    const avail = @min(@min(max_k, 2), search_len - start);
                    if (avail > 0) {
                        const recency: u32 = @intCast(@min(20, (idx * 20) / (search_len + 1)));
                        addOrUpdateBranch(tree, history[start .. start + avail], 30, recency);
                    }
                }
            }
        }

        // 4. Try 1-gram Markov follower match if no branches yet
        if (tree.count == 0 and history.len >= 2) {
            const last_tok = history[history.len - 1];
            const search_len = history.len - 1;
            var idx: usize = search_len;
            var matches: usize = 0;
            while (idx > 0 and matches < 8) {
                idx -= 1;
                if (history[idx] == last_tok and idx + 1 < search_len) {
                    matches += 1;
                    const avail = @min(@min(max_k, 2), search_len - (idx + 1));
                    if (avail > 0) {
                        const recency: u32 = @intCast(@min(20, (idx * 20) / (search_len + 1)));
                        addOrUpdateBranch(tree, history[idx + 1 .. idx + 1 + avail], 10, recency);
                    }
                }
            }
        }
    }

    pub fn find(history: []const u32, max_k: usize, out: []u32) usize {
        var tree = CandidateTree.init();
        buildTree(history, max_k, &tree);
        if (tree.best()) |b| {
            const avail = @min(b.len, out.len);
            @memcpy(out[0..avail], b.tokens[0..avail]);
            return avail;
        }
        return 0;
    }
};

test "PromptLookup finds continuation of matching 3-gram and 2-gram" {
    const hist = [_]u32{ 10, 20, 30, 40, 50, 60, 99, 10, 20, 30 };
    var draft: [4]u32 = undefined;
    const n3 = PromptLookup.find(&hist, 3, &draft);
    try std.testing.expectEqual(3, n3);
    try std.testing.expectEqualSlices(u32, &.{ 40, 50, 60 }, draft[0..3]);

    const hist2 = [_]u32{ 1, 2, 88, 99, 5, 6, 1, 2 };
    const n2 = PromptLookup.find(&hist2, 2, &draft);
    try std.testing.expectEqual(2, n2);
    try std.testing.expectEqualSlices(u32, &.{ 88, 99 }, draft[0..2]);

    const hist4 = [_]u32{ 10, 20, 30, 40, 50, 60, 70, 99, 10, 20, 30, 40 };
    const n4 = PromptLookup.find(&hist4, 3, &draft);
    try std.testing.expectEqual(3, n4);
    try std.testing.expectEqualSlices(u32, &.{ 50, 60, 70 }, draft[0..3]);
}

test "PromptLookup buildTree finds multiple candidate branches and ranks frequency" {
    // Pattern [1, 2]:
    // branch [8, 9] occurs once
    // branch [3, 4] occurs twice
    const hist = [_]u32{
        1, 2, 8, 9,
        100, 200,
        1, 2, 3, 4,
        300, 400,
        1, 2, 3, 4,
        500,
        1, 2,
    };
    var tree = CandidateTree.init();
    PromptLookup.buildTree(&hist, 4, &tree);
    try std.testing.expect(tree.count >= 2);
    const best = tree.best().?;
    // [3, 4] occurs twice -> higher score than [8, 9]
    try std.testing.expectEqualSlices(u32, &.{ 3, 4 }, best.tokens[0..best.len]);

    // Alternative branch starting with 8 is found
    const alt = tree.findAlternative(8);
    try std.testing.expect(alt != null);
    try std.testing.expectEqualSlices(u32, &.{ 8, 9 }, alt.?.tokens[0..alt.?.len]);

    const sec = tree.secondBest().?;
    try std.testing.expectEqualSlices(u32, &.{ 8, 9 }, sec.tokens[0..sec.len]);
}

const V8f = @Vector(8, f32);
const V8u = @Vector(8, u32);

pub const SampleResult = struct {
    tok: u32,
    margin: f32,
};

pub fn argmaxWithMargin(x: []const f32) SampleResult {
    if (x.len == 0) return .{ .tok = 0, .margin = 0 };
    if (x.len == 1) return .{ .tok = 0, .margin = 10.0 };
    if (x.len < 16) {
        var best: usize = 0;
        var second: usize = 1;
        if (x[1] > x[0]) {
            best = 1;
            second = 0;
        }
        for (x[2..], 2..) |v, i| {
            if (v > x[best]) {
                second = best;
                best = i;
            } else if (v > x[second]) {
                second = i;
            }
        }
        return .{ .tok = @intCast(best), .margin = x[best] - x[second] };
    }

    var max0: V8f = x[0..8].*;
    var max1: V8f = x[8..16].*;
    var idx0: V8u = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var idx1: V8u = .{ 8, 9, 10, 11, 12, 13, 14, 15 };
    var cur_idx0: V8u = idx0;
    var cur_idx1: V8u = idx1;
    const step: V8u = @splat(16);

    var i: usize = 16;
    while (i + 16 <= x.len) : (i += 16) {
        cur_idx0 +%= step;
        cur_idx1 +%= step;
        const v0: V8f = x[i..][0..8].*;
        const v1: V8f = x[i + 8 ..][0..8].*;
        const m0 = v0 > max0;
        const m1 = v1 > max1;
        max0 = @select(f32, m0, v0, max0);
        idx0 = @select(u32, m0, cur_idx0, idx0);
        max1 = @select(f32, m1, v1, max1);
        idx1 = @select(u32, m1, cur_idx1, idx1);
    }

    var best_val: f32 = -std.math.inf(f32);
    var second_val: f32 = -std.math.inf(f32);
    var best_idx: u32 = 0;

    inline for (0..8) |j| {
        const v0 = max0[j];
        if (v0 > best_val) {
            second_val = best_val;
            best_val = v0;
            best_idx = idx0[j];
        } else if (v0 > second_val) {
            second_val = v0;
        }
        const v1 = max1[j];
        if (v1 > best_val) {
            second_val = best_val;
            best_val = v1;
            best_idx = idx1[j];
        } else if (v1 > second_val) {
            second_val = v1;
        }
    }

    while (i < x.len) : (i += 1) {
        if (x[i] > best_val) {
            second_val = best_val;
            best_val = x[i];
            best_idx = @intCast(i);
        } else if (x[i] > second_val) {
            second_val = x[i];
        }
    }

    const margin = if (second_val > -std.math.inf(f32)) best_val - second_val else 10.0;
    return .{ .tok = best_idx, .margin = margin };
}

pub fn argmax(x: []const f32) u32 {
    return argmaxWithMargin(x).tok;
}

test "argmax correctness" {
    var arr: [1000]f32 = @splat(-100.0);
    arr[42] = 50.0;
    try std.testing.expectEqual(@as(u32, 42), argmax(&arr));

    arr[42] = -100.0;
    arr[999] = 100.0;
    try std.testing.expectEqual(@as(u32, 999), argmax(&arr));

    arr[999] = -100.0;
    arr[0] = 12.0;
    try std.testing.expectEqual(@as(u32, 0), argmax(&arr));

    arr[0] = -100.0;
    arr[7] = 88.0;
    try std.testing.expectEqual(@as(u32, 7), argmax(&arr));
}

test "DRY sampler penalty and sequence breaker" {
    const gpa = std.testing.allocator;
    var smp = try Sampler.init(gpa, 100, .{
        .dry_multiplier = 0.8,
        .dry_base = 2.0,
        .dry_allowed_length = 2,
        .dry_range = 1024,
    });
    defer smp.deinit(gpa);

    // Sequence: [1, 2, 3, 4, 1, 2]
    // Suffix [1, 2] matches earlier [1, 2], whose continuation is 3.
    const seq1 = [_]u32{ 1, 2, 3, 4, 1, 2 };
    for (seq1) |t| smp.accept(t);

    var logits: [100]f32 = @splat(10.0);
    smp.applyDry(&logits);

    // Token 3 should be penalized by 0.8 * (2.0^0) = 0.8 -> 9.2
    try std.testing.expectApproxEqAbs(@as(f32, 9.2), logits[3], 1e-4);
    // Other tokens like 4 or 5 should remain 10.0
    try std.testing.expectEqual(@as(f32, 10.0), logits[4]);
    try std.testing.expectEqual(@as(f32, 10.0), logits[5]);

    // Now accept token 3 so sequence is [1, 2, 3, 4, 1, 2, 3]
    smp.accept(3);
    var logits2: [100]f32 = @splat(10.0);
    smp.applyDry(&logits2);

    // Suffix [1, 2, 3] matches earlier [1, 2, 3] of length 3, continuation is 4.
    // Penalty is 0.8 * (2.0^(3-2)) = 1.6 -> 8.4
    try std.testing.expectApproxEqAbs(@as(f32, 8.4), logits2[4], 1e-4);

    // Test sequence breaker: mark token 4 as breaker
    smp.addBreaker(4);
    var logits3: [100]f32 = @splat(10.0);
    smp.applyDry(&logits3);
    // Token 4 is a breaker, so it should NOT be penalized
    try std.testing.expectEqual(@as(f32, 10.0), logits3[4]);
}
