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
    /// Whether to dynamically tune lookahead based on real-time acceptance rate.
    draft_adaptive: bool = true,
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
    prompt_cache_save: ?[]const u8 = null,

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

    pub const CacheHeader = extern struct {
        magic: [8]u8 = "ZINFCACH".*,
        version: u32 = 1,
        arch: u32,
        n_layers: u32,
        dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        vocab: u32,
        kv_format: u32,
        pos: u64,
        n_tokens: u64,
        conv_len: u64,
    };

    pub fn saveCache(g: *Gen, path: []const u8) !void {
        const file = try std.Io.Dir.cwd().createFile(g.io, path, .{});
        defer file.close(g.io);
        var buf: [64 * 1024]u8 = undefined;
        var writer = file.writer(g.io, &buf);

        const st = g.st;
        const c = &st.model.cfg;
        const hdr: CacheHeader = .{
            .arch = @intFromEnum(c.arch),
            .n_layers = @intCast(c.n_layers),
            .dim = @intCast(c.dim),
            .n_heads = @intCast(c.n_heads),
            .n_kv_heads = @intCast(c.n_kv_heads),
            .head_dim = @intCast(c.head_dim),
            .vocab = @intCast(c.vocab),
            .kv_format = @intFromEnum(st.cache_type),
            .pos = @intCast(st.pos),
            .n_tokens = @intCast(g.cached.items.len),
            .conv_len = @intCast(st.conv_state.len),
        };
        var hasher = std.hash.XxHash64.init(0);
        hasher.update(std.mem.asBytes(&hdr));
        try writer.interface.writeAll(std.mem.asBytes(&hdr));

        if (g.cached.items.len > 0) {
            const slice = std.mem.sliceAsBytes(g.cached.items);
            hasher.update(slice);
            try writer.interface.writeAll(slice);
        }
        if (st.conv_state.len > 0) {
            const slice = std.mem.sliceAsBytes(st.conv_state);
            hasher.update(slice);
            try writer.interface.writeAll(slice);
        }

        const tb: usize = switch (st.cache_type) {
            .f16 => c.head_dim * 32,
            .q8_0 => c.head_dim * 17,
            .q4_0 => c.head_dim * 9,
        };
        for (st.model.layers) |*L| {
            switch (L.mixer) {
                .attention => |*A| {
                    const is_swa = A.swa and c.swa_window > 0;
                    const max_tiles = st.slot_tiles[A.kv_slot];
                    const tiles = if (st.pos == 0) 0 else if (is_swa) @min(max_tiles, (st.pos + model_mod.kv_tile - 1) / model_mod.kv_tile) else (st.pos + model_mod.kv_tile - 1) / model_mod.kv_tile;
                    const bytes_per_head = tiles * tb;
                    for (0..A.n_kv_heads) |kh| {
                        const base = st.cacheIndex(A.kv_slot, kh);
                        const k_slice = st.k_cache[base .. base + bytes_per_head];
                        const v_slice = st.v_cache[base .. base + bytes_per_head];
                        hasher.update(k_slice);
                        try writer.interface.writeAll(k_slice);
                        hasher.update(v_slice);
                        try writer.interface.writeAll(v_slice);
                    }
                },
                .shortconv => {},
            }
        }
        const checksum = hasher.final();
        try writer.interface.writeAll(std.mem.asBytes(&checksum));
        try writer.interface.flush();
    }

    pub fn loadCache(g: *Gen, path: []const u8) !void {
        const file = try std.Io.Dir.cwd().openFile(g.io, path, .{});
        defer file.close(g.io);
        var buf: [64 * 1024]u8 = undefined;
        var reader = file.reader(g.io, &buf);

        var hdr: CacheHeader = undefined;
        try reader.interface.readSliceAll(std.mem.asBytes(&hdr));

        if (!std.mem.eql(u8, &hdr.magic, "ZINFCACH") or hdr.version != 1) return error.InvalidCache;

        var hasher = std.hash.XxHash64.init(0);
        hasher.update(std.mem.asBytes(&hdr));

        const st = g.st;
        const c = &st.model.cfg;
        if (hdr.arch != @intFromEnum(c.arch) or
            hdr.n_layers != c.n_layers or
            hdr.dim != c.dim or
            hdr.n_heads != c.n_heads or
            hdr.n_kv_heads != c.n_kv_heads or
            hdr.head_dim != c.head_dim or
            hdr.vocab != c.vocab or
            hdr.kv_format != @intFromEnum(st.cache_type))
        {
            return error.IncompatibleCache;
        }

        if (hdr.pos > st.ctx) return error.ContextFull;

        g.reset();
        try g.cached.resize(g.gpa, @intCast(hdr.n_tokens));
        if (hdr.n_tokens > 0) {
            const slice = std.mem.sliceAsBytes(g.cached.items);
            try reader.interface.readSliceAll(slice);
            hasher.update(slice);
        }
        if (hdr.conv_len > 0) {
            if (hdr.conv_len != st.conv_state.len) return error.IncompatibleCache;
            const slice = std.mem.sliceAsBytes(st.conv_state);
            try reader.interface.readSliceAll(slice);
            hasher.update(slice);
        }

        const tb: usize = switch (st.cache_type) {
            .f16 => c.head_dim * 32,
            .q8_0 => c.head_dim * 17,
            .q4_0 => c.head_dim * 9,
        };
        const pos: usize = @intCast(hdr.pos);
        for (st.model.layers) |*L| {
            switch (L.mixer) {
                .attention => |*A| {
                    const is_swa = A.swa and c.swa_window > 0;
                    const max_tiles = st.slot_tiles[A.kv_slot];
                    const tiles = if (pos == 0) 0 else if (is_swa) @min(max_tiles, (pos + model_mod.kv_tile - 1) / model_mod.kv_tile) else (pos + model_mod.kv_tile - 1) / model_mod.kv_tile;
                    const bytes_per_head = tiles * tb;
                    for (0..A.n_kv_heads) |kh| {
                        const base = st.cacheIndex(A.kv_slot, kh);
                        const k_slice = st.k_cache[base .. base + bytes_per_head];
                        const v_slice = st.v_cache[base .. base + bytes_per_head];
                        try reader.interface.readSliceAll(k_slice);
                        hasher.update(k_slice);
                        try reader.interface.readSliceAll(v_slice);
                        hasher.update(v_slice);
                    }
                },
                .shortconv => {},
            }
        }

        var stored_checksum: u64 = undefined;
        reader.interface.readSliceAll(std.mem.asBytes(&stored_checksum)) catch |e| switch (e) {
            error.EndOfStream => {
                st.pos = pos;
                if (g.draft_st) |d| d.reset();
                return;
            },
            else => return e,
        };
        if (stored_checksum != hasher.final()) return error.InvalidCacheChecksum;
        st.pos = pos;
        if (g.draft_st) |d| d.reset();
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
            if (e != error.ContextFull) g.reset();
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

    pub const AdaptiveDraft = struct {
        cur_k: usize,
        max_k: usize,
        min_k: usize = 1,
        consecutive_rejections: usize = 0,
        cooldown_steps: usize = 0,
        last_margin: f32 = 2.0,

        pub fn init(max_k: usize) AdaptiveDraft {
            return .{
                .cur_k = @min(max_k, 3),
                .max_k = max_k,
                .last_margin = 2.0,
            };
        }

        pub fn recordMargin(self: *AdaptiveDraft, margin: f32) void {
            self.last_margin = margin;
        }

        /// Returns target draft length for this round based on both acceptance history
        /// and real-time entropy margin of the logits.
        pub fn nextDraft(self: *AdaptiveDraft) usize {
            if (self.cooldown_steps > 0) {
                self.cooldown_steps -= 1;
                return 0;
            }
            // Entropy-adaptive scaling:
            // High margin (low entropy): model is very confident, expand lookahead up to max_k.
            // Low margin (high entropy): model is uncertain, contract to avoid wasted speculation.
            var target_k = self.cur_k;
            if (self.last_margin >= 3.0) {
                target_k = @min(self.max_k, @max(self.cur_k, 4));
            } else if (self.last_margin >= 1.8) {
                target_k = @min(self.max_k, @max(self.cur_k, 2));
            } else if (self.last_margin < 0.9) {
                target_k = @min(target_k, 1);
            }
            return target_k;
        }

        /// Feedback update after verification round.
        pub fn update(self: *AdaptiveDraft, accepted: usize, drafted: usize, margin: ?f32) void {
            if (margin) |m| self.last_margin = m;
            if (drafted == 0) return;

            if (accepted == drafted) {
                self.consecutive_rejections = 0;
                if (self.cur_k < self.max_k) self.cur_k += 1;
            } else if (accepted > 0) {
                self.consecutive_rejections = 0;
                if (accepted * 2 < drafted and self.cur_k > self.min_k) {
                    self.cur_k -= 1;
                }
            } else {
                self.consecutive_rejections += 1;
                if (self.cur_k > self.min_k) {
                    self.cur_k -= 1;
                }
                if (self.consecutive_rejections >= 2) {
                    self.cooldown_steps = 2;
                    self.consecutive_rejections = 0;
                    self.cur_k = @min(self.max_k, 2);
                }
            }
        }
    };

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

        if (g.prompt_cache_save) |save_path| {
            // Populate cached prefix with prompt before saving
            g.cached.shrinkRetainingCapacity(keep);
            try g.cached.appendSlice(g.gpa, feed);
            try g.saveCache(save_path);
            g.prompt_cache_save = null;
        }

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
        var adaptive: AdaptiveDraft = .init(g.draft_k);

        while (n_run < n and g.st.pos < g.st.ctx) {
            const current_max_k = if (g.draft_adaptive) adaptive.nextDraft() else g.draft_k;
            const max_draft = @min(current_max_k, max_spec_draft);
            const d = g.draft_st;
            // Invariant at round start: the draft cache ends just before
            // cur_tok, mirroring the target cache.
            const k = blk: {
                if (max_draft == 0) break :blk 0;
                if (d) |dst| {
                    if (g.st.pos + max_draft + 1 > g.st.ctx or dst.pos + max_draft + 1 > dst.ctx)
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
                if (g.st.pos + max_draft + 2 <= g.st.ctx)
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
                const res = g.smp.sampleWithMargin(l);
                const next_tok = res.tok;
                if (g.draft_adaptive) adaptive.recordMargin(res.margin);
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
                var last_margin: f32 = 2.0;

                while (accepted_count < k) : (accepted_count += 1) {
                    const l = all_logits[accepted_count * vocab ..][0..vocab];
                    const res = g.smp.sampleWithMargin(l);
                    const pred = res.tok;
                    last_margin = res.margin;
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
                    const res = g.smp.sampleWithMargin(l);
                    next_cur = res.tok;
                    last_margin = res.margin;
                }

                if (g.draft_adaptive) adaptive.update(accepted_count, k, last_margin);

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

test "reusablePrefix" {
    var g: Gen = undefined;
    var fake_st: model_mod.State = undefined;
    fake_st.conv_state = &.{};
    g.st = &fake_st;
    g.cached = .empty;
    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(std.testing.allocator);
    try list.appendSlice(std.testing.allocator, &.{ 1, 2, 3, 4, 5 });
    g.cached = list;
    g.st.pos = 5;

    // Prompt that matches completely:
    // reusablePrefix returns prompt.len - 1 so at least one token is fed for logits
    try std.testing.expectEqual(@as(usize, 4), g.reusablePrefix(&.{ 1, 2, 3, 4, 5 }));

    // Prompt with extension:
    try std.testing.expectEqual(@as(usize, 5), g.reusablePrefix(&.{ 1, 2, 3, 4, 5, 6, 7 }));

    // Prompt with partial divergence:
    try std.testing.expectEqual(@as(usize, 3), g.reusablePrefix(&.{ 1, 2, 3, 99, 100 }));

    // Prompt with completely different start:
    try std.testing.expectEqual(@as(usize, 0), g.reusablePrefix(&.{ 9, 8, 7 }));

    // Empty prompt:
    try std.testing.expectEqual(@as(usize, 0), g.reusablePrefix(&.{}));
}

test "saveCache and loadCache roundtrip" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    var model: model_mod.Model = undefined;
    model.cfg = .{
        .arch = .llama,
        .dim = 64,
        .n_layers = 1,
        .n_heads = 2,
        .n_kv_heads = 1,
        .head_dim = 32,
        .q_dim = 64,
        .kv_dim = 32,
        .ffn_dim = 128,
        .vocab = 1000,
        .eps = 1e-5,
        .rope_base = 10000,
        .rope_scale = 1,
        .n_rot = 32,
        .rope_neox = true,
        .ctx_train = 512,
    };
    var layer: model_mod.Layer = undefined;
    layer.mixer = .{
        .attention = .{
            .wq = undefined,
            .wk = undefined,
            .wv = undefined,
            .wo = undefined,
            .gate = null,
            .bq = null,
            .bk = null,
            .bv = null,
            .q_norm = null,
            .k_norm = null,
            .swa = false,
            .n_kv_heads = 1,
            .kv_slot = 0,
        },
    };
    model.layers = (&layer)[0..1];
    model.n_attn_layers = 1;
    model.n_conv_layers = 0;

    var st: model_mod.State = undefined;
    st.model = &model;
    st.ctx = 256;
    st.cache_type = .f16;
    st.pos = 32; // 2 tiles of 16 tokens
    st.conv_state = &.{};
    st.conv_checkpoint = &.{};

    var slot_tiles = [_]usize{ 16 };
    var slot_strides = [_]usize{ 16 * 32 * 32 };
    var slot_offsets = [_]usize{ 0 };
    st.slot_tiles = &slot_tiles;
    st.slot_strides = &slot_strides;
    st.slot_offsets = &slot_offsets;

    const cache_bytes = 16 * 32 * 32;
    var k_cache: [cache_bytes]u8 align(32) = undefined;
    var v_cache: [cache_bytes]u8 align(32) = undefined;
    for (&k_cache, 0..) |*b, i| b.* = @truncate(i * 3 + 1);
    for (&v_cache, 0..) |*b, i| b.* = @truncate(i * 7 + 5);
    st.k_cache = &k_cache;
    st.v_cache = &v_cache;

    var cached_toks: std.ArrayList(u32) = .empty;
    defer cached_toks.deinit(gpa);
    for (0..32) |i| try cached_toks.append(gpa, @intCast(100 + i));

    var smp: sampler_mod.Sampler = undefined;
    smp.n_recent = 0;

    var g: Gen = undefined;
    g.gpa = gpa;
    g.io = io;
    g.st = &st;
    g.smp = &smp;
    g.cached = cached_toks;
    g.draft_st = null;
    g.snaps = &.{};
    g.n_snaps = 0;

    const test_cache_path = "test_prompt.cache";
    defer std.Io.Dir.cwd().deleteFile(io, test_cache_path) catch {};

    try g.saveCache(test_cache_path);

    // Verify cache file exists
    const file = try std.Io.Dir.cwd().openFile(io, test_cache_path, .{});
    file.close(io);

    // Prepare fresh state to load into
    var st2: model_mod.State = undefined;
    st2.model = &model;
    st2.ctx = 256;
    st2.cache_type = .f16;
    st2.pos = 0;
    st2.conv_state = &.{};
    st2.conv_checkpoint = &.{};
    st2.slot_tiles = &slot_tiles;
    st2.slot_strides = &slot_strides;
    st2.slot_offsets = &slot_offsets;

    var k_cache2: [cache_bytes]u8 align(32) = undefined;
    var v_cache2: [cache_bytes]u8 align(32) = undefined;
    @memset(&k_cache2, 0);
    @memset(&v_cache2, 0);
    st2.k_cache = &k_cache2;
    st2.v_cache = &v_cache2;

    var g2: Gen = undefined;
    g2.gpa = gpa;
    g2.io = io;
    g2.st = &st2;
    g2.smp = &smp;
    g2.cached = .empty;
    defer g2.cached.deinit(gpa);
    g2.draft_st = null;
    g2.snaps = &.{};
    g2.n_snaps = 0;

    try g2.loadCache(test_cache_path);

    try std.testing.expectEqual(@as(usize, 32), st2.pos);
    try std.testing.expectEqual(@as(usize, 32), g2.cached.items.len);
    for (0..32) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(100 + i)), g2.cached.items[i]);
    }

    // 2 tiles * 32 head_dim * 32 bytes = 2048 bytes
    const valid_bytes = 2 * 32 * 32;
    try std.testing.expectEqualSlices(u8, k_cache[0..valid_bytes], k_cache2[0..valid_bytes]);
    try std.testing.expectEqualSlices(u8, v_cache[0..valid_bytes], v_cache2[0..valid_bytes]);

    // Incompatible configuration check
    var model_incompat = model;
    model_incompat.cfg.dim = 128;
    st2.model = &model_incompat;
    try std.testing.expectError(error.IncompatibleCache, g2.loadCache(test_cache_path));

    // Exceeding context limit check
    st2.model = &model;
    st2.ctx = 16;
    try std.testing.expectError(error.ContextFull, g2.loadCache(test_cache_path));
}

test "AdaptiveDraft feedback control" {
    var ad = Gen.AdaptiveDraft.init(5);
    try std.testing.expectEqual(@as(usize, 3), ad.nextDraft());

    // Perfect round: 3/3 accepted -> grows to 4
    ad.update(3, 3, 2.5);
    try std.testing.expectEqual(@as(usize, 4), ad.nextDraft());

    // Perfect round: 4/4 accepted -> grows to 5 (max)
    ad.update(4, 4, 2.5);
    try std.testing.expectEqual(@as(usize, 5), ad.nextDraft());

    // Partial success: 3/5 accepted -> stays 5
    ad.update(3, 5, 2.5);
    try std.testing.expectEqual(@as(usize, 5), ad.nextDraft());

    // Low acceptance: 1/5 accepted (< 5/2) -> drops to 4
    ad.update(1, 5, 2.5);
    try std.testing.expectEqual(@as(usize, 4), ad.nextDraft());

    // 0 accepted -> drops to 3
    ad.update(0, 4, 2.5);
    try std.testing.expectEqual(@as(usize, 3), ad.nextDraft());

    // 0 accepted second time in a row -> triggers cooldown for 2 steps!
    ad.update(0, 3, 2.5);
    try std.testing.expectEqual(@as(usize, 0), ad.nextDraft()); // cooldown step 1
    try std.testing.expectEqual(@as(usize, 0), ad.nextDraft()); // cooldown step 2
    try std.testing.expectEqual(@as(usize, 2), ad.nextDraft()); // resumes with cautious probe k=2
}
