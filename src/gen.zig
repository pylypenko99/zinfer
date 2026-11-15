//! Generation loop shared by the CLI and the HTTP server: sampling plus
//! n-gram or draft-model speculative decoding, with timing stats.
const std = @import("std");
const model_mod = @import("model.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const sampler_mod = @import("sampler.zig");

/// Monotonic stopwatch on the Io awake clock.
pub const Timer = struct {
    io: std.Io,
    last: std.Io.Timestamp,

    pub fn start(io: std.Io) Timer {
        return .{ .io = io, .last = std.Io.Clock.awake.now(io) };
    }

    pub fn read(t: *const Timer) u64 {
        return @intCast(t.last.durationTo(std.Io.Clock.awake.now(t.io)).nanoseconds);
    }

    pub fn lap(t: *Timer) u64 {
        const now = std.Io.Clock.awake.now(t.io);
        const ns: u64 = @intCast(t.last.durationTo(now).nanoseconds);
        t.last = now;
        return ns;
    }
};

pub const Stats = struct {
    n_prompt: usize = 0,
    /// Prompt tokens reused from the cache instead of being processed.
    n_cached: usize = 0,
    n_gen: usize = 0,
    n_drafted: usize = 0,
    n_accepted: usize = 0,
    t_prompt_ns: u64 = 0,
    t_gen_ns: u64 = 0,
};

pub const Gen = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    st: *model_mod.State,
    tok: *const Tokenizer,
    smp: *sampler_mod.Sampler,
    /// Speculative lookahead: n-gram prompt lookup, or the draft model if set.
    draft_k: usize = 0,
    /// Draft model state and sampler; between rounds its cache mirrors `st`.
    draft_st: ?*model_mod.State = null,
    draft_smp: ?*sampler_mod.Sampler = null,
    stats: Stats = .{},
    /// Tokens in the target cache, in order (`st.pos` of them).
    cached: std.ArrayList(u32) = .empty,
    /// Short-conv states saved by `runPrompt` at the end of the last prompt:
    /// entry i is the state at position `snap_pos + i`. Attention caches can
    /// be cut at any position, short-conv state only where it was saved.
    snaps: []f32 = &.{},
    snap_pos: usize = 0,
    n_snaps: usize = 0,

    pub fn deinit(g: *Gen) void {
        g.cached.deinit(g.gpa);
        g.gpa.free(g.snaps);
    }

    /// Forget the context of both models and the repeat-penalty history.
    pub fn reset(g: *Gen) void {
        g.st.reset();
        if (g.draft_st) |d| d.reset();
        g.smp.n_recent = 0;
        g.cached.clearRetainingCapacity();
        g.n_snaps = 0;
    }

    /// Feed `input` after the current context, then sample up to `n` tokens,
    /// passing each token's bytes to `sink.emit`. Returns the end-of-generation
    /// token if one was sampled (it is not yet in the KV cache).
    pub fn run(g: *Gen, input: []const u32, n: usize, sink: anytype) !?u32 {
        return g.generate(input, 0, false, n, sink);
    }

    /// `reset` followed by `run(prompt)` for a request that carries the whole
    /// conversation, except that the prefix shared with the cached context is
    /// kept and only the rest of `prompt` is processed.
    pub fn runPrompt(g: *Gen, prompt: []const u32, n: usize, sink: anytype) !?u32 {
        const keep = g.reusablePrefix(prompt);
        g.truncate(keep);
        g.smp.n_recent = 0;
        return g.generate(prompt, keep, g.st.conv_state.len > 0, n, sink);
    }

    /// Longest cached prefix of `prompt` both caches can be cut back to. At
    /// least one prompt token stays to be processed: its logits start the
    /// generation.
    fn reusablePrefix(g: *const Gen, prompt: []const u32) usize {
        if (prompt.len == 0) return 0;
        const shared = std.mem.indexOfDiff(u32, g.cached.items, prompt) orelse prompt.len;
        const limit = @min(shared, prompt.len - 1);
        if (g.st.conv_state.len == 0 or limit == g.st.pos) return limit;
        if (g.n_snaps == 0 or limit < g.snap_pos) return 0;
        return g.snap_pos + @min(limit - g.snap_pos, g.n_snaps - 1);
    }

    /// Cut both caches back to `keep` tokens, a position from `reusablePrefix`.
    fn truncate(g: *Gen, keep: usize) void {
        if (keep == g.st.pos) return;
        if (keep == 0) return g.reset();
        const sz = g.st.conv_state.len;
        if (sz == 0) {
            g.st.truncate(keep, &.{});
        } else {
            g.st.truncate(keep, g.snaps[(keep - g.snap_pos) * sz ..][0..sz]);
            // Later snapshots belong to the discarded continuation.
            g.n_snaps = keep - g.snap_pos + 1;
        }
        // The draft mirrors the target cache and has no short-conv layers.
        if (g.draft_st) |d| d.truncate(keep, &.{});
        g.cached.shrinkRetainingCapacity(keep);
    }

    /// Feed `prompt[keep..]` to a cache that ends with `prompt[0..keep]`, then
    /// sample as `run` does. With `snapshot`, the short-conv states at the end
    /// of the prompt are saved for the next `runPrompt`.
    fn generate(g: *Gen, prompt: []const u32, keep: usize, snapshot: bool, n: usize, sink: anytype) !?u32 {
        std.debug.assert(g.cached.items.len == g.st.pos and keep < prompt.len);
        const base = g.st.pos - keep;
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(g.gpa);
        const eog = g.decode(prompt, keep, snapshot, n, sink, &history) catch |e| {
            // An interrupted round can leave unverified drafts in the caches.
            g.reset();
            return e;
        };
        // Tokens enter the cache in `history` order, so it holds a prefix of it.
        g.cached.shrinkRetainingCapacity(base);
        g.cached.appendSlice(g.gpa, history.items[0 .. g.st.pos - base]) catch |e| {
            g.reset();
            return e;
        };
        return eog;
    }

    /// Prefill with the last few tokens as one small batch: its short-conv
    /// checkpoints become the snapshots `runPrompt` can cut back to. Clients
    /// diverge from the cached context right there: a prompt ends with the
    /// assistant header (and `<think>`), the next request with the stored reply.
    fn prefillSnapshots(g: *Gen, feed: []const u32) ![]f32 {
        if (g.st.pos + feed.len > g.st.ctx) return error.ContextFull;
        const tail = @min(feed.len, model_mod.max_spec);
        if (feed.len > tail) _ = try g.st.forwardN(feed[0 .. feed.len - tail], 0);
        const logits = try g.st.forward(feed[feed.len - tail ..]);
        const sz = g.st.conv_state.len;
        if (g.snaps.len == 0) g.snaps = try g.gpa.alloc(f32, (model_mod.max_spec + 1) * sz);
        for (0..tail + 1) |t| @memcpy(g.snaps[t * sz ..][0..sz], g.st.convCheckpointState(t));
        g.snap_pos = g.st.pos - tail;
        g.n_snaps = tail + 1;
        return logits;
    }

    fn decode(g: *Gen, prompt: []const u32, keep: usize, snapshot: bool, n: usize, sink: anytype, history: *std.ArrayList(u32)) !?u32 {
        var timer: Timer = .start(g.io);
        const feed = prompt[keep..];
        const logits = if (snapshot) try g.prefillSnapshots(feed) else try g.st.forward(feed);
        // The draft mirrors the target cache; it needs no logits for the input.
        if (g.draft_st) |dst| _ = try dst.forwardN(feed, 0);
        g.stats.t_prompt_ns += timer.lap();
        g.stats.n_prompt += feed.len;
        g.stats.n_cached += keep;
        for (prompt) |t| g.smp.accept(t);
        try history.appendSlice(g.gpa, prompt);

        // Tokens emitted by this call: `n` limits every call on its own.
        var n_run: usize = 0;
        var cur_tok = g.smp.sample(logits);
        if (g.tok.isEog(cur_tok)) {
            g.stats.t_gen_ns += timer.lap();
            return cur_tok;
        }
        try g.take(sink, history, cur_tok, &n_run);

        const max_spec_draft = model_mod.max_spec - 1;
        var draft_buf: [max_spec_draft]u32 = undefined;
        var cand_buf: [model_mod.max_spec]u32 = undefined;

        while (n_run < n and g.st.pos < g.st.ctx) {
            const max_draft = @min(g.draft_k, max_spec_draft);
            const d = g.draft_st;
            // Invariant at round start: the draft cache ends just before
            // cur_tok, mirroring the target cache.
            const k = blk: {
                if (d) |dst| {
                    if (max_draft == 0 or g.st.pos + max_draft + 1 > g.st.ctx or dst.pos + max_draft + 1 > dst.ctx)
                        break :blk 0;
                    // Autoregressive draft: feed cur_tok, then each sampled
                    // token. Stop early on EOG (the target rejects it anyway).
                    var l = try dst.forward(&.{cur_tok});
                    var produced: usize = 0;
                    while (produced < max_draft) : (produced += 1) {
                        const t = g.draft_smp.?.sample(l);
                        if (g.tok.isEog(t)) break;
                        draft_buf[produced] = t;
                        if (produced + 1 < max_draft) l = try dst.forward(&.{t});
                    }
                    break :blk produced;
                }
                if (max_draft > 0 and g.st.pos + max_draft + 2 <= g.st.ctx)
                    break :blk sampler_mod.PromptLookup.find(history.items, max_draft, &draft_buf);
                break :blk 0;
            };

            if (k == 0) {
                const l = try g.st.forward(&.{cur_tok});
                // Keep the draft level with the target unless its round above
                // already consumed cur_tok (first draft sample was EOG).
                if (d) |dst| {
                    if (dst.pos < g.st.pos) _ = try dst.forwardN(&.{cur_tok}, 0);
                }
                const next_tok = g.smp.sample(l);
                if (g.tok.isEog(next_tok)) {
                    g.stats.t_gen_ns += timer.lap();
                    return next_tok;
                }
                try g.take(sink, history, next_tok, &n_run);
                cur_tok = next_tok;
            } else {
                g.stats.n_drafted += k;
                cand_buf[0] = cur_tok;
                @memcpy(cand_buf[1 .. 1 + k], draft_buf[0..k]);
                const total_cand = 1 + k;

                const all_logits = try g.st.forwardN(cand_buf[0..total_cand], total_cand);
                const vocab = g.st.model.cfg.vocab;

                var accepted_count: usize = 0;
                var next_cur: ?u32 = null;

                while (accepted_count < k) : (accepted_count += 1) {
                    const l = all_logits[accepted_count * vocab ..][0..vocab];
                    const pred = g.smp.sample(l);
                    if (pred == draft_buf[accepted_count] and !g.tok.isEog(pred) and n_run < n) {
                        try g.take(sink, history, pred, &n_run);
                        g.stats.n_accepted += 1;
                    } else {
                        next_cur = pred;
                        break;
                    }
                }

                if (next_cur == null) {
                    const l = all_logits[k * vocab ..][0..vocab];
                    next_cur = g.smp.sample(l);
                }

                const kept_in_cache = 1 + accepted_count;
                g.st.rewind(kept_in_cache, total_cand);
                // The draft cache holds cur_tok plus every drafted token it was
                // fed: all but the last one, or all of them when it stopped at
                // EOG. Accepted tokens equal drafted ones, so keep the target's
                // prefix, feeding the single missing token if all were accepted.
                if (d) |dst| {
                    if (dst.pos < g.st.pos) {
                        _ = try dst.forwardN(draft_buf[k - 1 .. k], 0);
                    } else dst.pos = g.st.pos;
                }

                if (n_run >= n) break;
                const next_tok = next_cur.?;
                if (g.tok.isEog(next_tok)) {
                    g.stats.t_gen_ns += timer.lap();
                    return next_tok;
                }
                try g.take(sink, history, next_tok, &n_run);
                cur_tok = next_tok;
            }
        }
        g.stats.t_gen_ns += timer.lap();
        return null;
    }

    /// Emit a sampled token and add it to the context history.
    fn take(g: *Gen, sink: anytype, history: *std.ArrayList(u32), t: u32, n_run: *usize) !void {
        try sink.emit(g.tok.piece(t));
        g.smp.accept(t);
        try history.append(g.gpa, t);
        n_run.* += 1;
        g.stats.n_gen += 1;
    }

    /// Print the stats line to stderr.
    pub fn report(g: *const Gen) void {
        const s = g.stats;
        const ms = struct {
            fn f(ns: u64) f64 {
                return @as(f64, @floatFromInt(ns)) / 1e6;
            }
        }.f;
        const rate = struct {
            fn f(n: usize, ns: u64) f64 {
                return if (ns == 0) 0 else @as(f64, @floatFromInt(n)) * 1e9 / @as(f64, @floatFromInt(ns));
            }
        }.f;
        std.debug.print("prompt: {d} tok in {d:.1} ms ({d:.2} tok/s)", .{ s.n_prompt, ms(s.t_prompt_ns), rate(s.n_prompt, s.t_prompt_ns) });
        if (s.n_cached > 0) std.debug.print(" + {d} cached", .{s.n_cached});
        std.debug.print(" | gen: {d} tok in {d:.1} ms ({d:.2} tok/s)", .{ s.n_gen, ms(s.t_gen_ns), rate(s.n_gen, s.t_gen_ns) });
        if (s.n_drafted > 0) {
            std.debug.print(" | spec: {d}/{d} accepted ({d:.1}%)\n", .{
                s.n_accepted,
                s.n_drafted,
                @as(f64, @floatFromInt(s.n_accepted)) * 100.0 / @as(f64, @floatFromInt(s.n_drafted)),
            });
        } else {
            std.debug.print("\n", .{});
        }
    }
};
