//! Transformer weights and the forward pass (Llama / Qwen2 / Qwen3 families).
//!
//! One code path serves both prompt processing and generation: `forward`
//! takes a batch of tokens, so every weight row is streamed from RAM once per
//! batch instead of once per token. Matmul weights are repacked at load time
//! into 8-row interleaved blocks (see repack.zig) held in huge-page memory.
//! Threads split output rows dynamically; QKV and gate/up projections are
//! fused into single dispatches.
const std = @import("std");
const posix = std.posix;
const gguf = @import("gguf.zig");
const quant = @import("quant.zig");
const repack = @import("repack.zig");
const pool_mod = @import("pool.zig");
const Pool = pool_mod.Pool;
const Chunks = pool_mod.Chunks;
const GgmlType = gguf.GgmlType;
const Allocator = std.mem.Allocator;

pub const rope_mod = @import("rope.zig");
pub const attention_mod = @import("attention.zig");
pub const ffn_mod = @import("ffn.zig");

pub const rope = rope_mod;
pub const attention = attention_mod;
pub const ffn = ffn_mod;

pub const Arch = enum { llama, qwen2, qwen3, lfm2, lfm3, gemma3, gemma4, qwen2moe, mixtral, deepseek2 };
pub const CacheType = attention.CacheType;
pub const max_layers = 256;
pub const max_active_experts = 256;

// Re-exports from decomposed modules for backwards-compatibility
pub const kv_tile = attention.kv_tile;
pub const max_q = attention.max_q;
pub const max_head_dim = attention.max_head_dim;
pub const max_spec = attention.max_spec;
pub const max_flash_chunks = attention.max_flash_chunks;
pub const PartialStats = attention.PartialStats;
pub const fwhtSimd = attention.fwhtSimd;
pub const pmmScaleBlock32_q4 = attention.pmmScaleBlock32_q4;
pub const pmmScaleBlock32_q2 = attention.pmmScaleBlock32_q2;
pub const quantBlock32 = attention.quantBlock32;
pub const quantBlock32_q4 = attention.quantBlock32_q4;
pub const quantBlock32_q2 = attention.quantBlock32_q2;
pub const attendBlock = attention.attendBlock;
pub const attendBlockChunk = attention.attendBlockChunk;

pub const expV = ffn.expV;
pub const siluMulV = ffn.siluMulV;
pub const geluMulV = ffn.geluMulV;
pub const silu_mul = ffn.silu_mul;
pub const gelu_mul = ffn.gelu_mul;
pub const routeTopK = ffn.routeTopK;

pub const ropeFreqs = rope.ropeFreqs;
pub const ntkAwareBase = rope.ntkAwareBase;

pub const Config = struct {
    arch: Arch,
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    /// Largest KV head count of any attention layer.
    n_kv_heads: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    ffn_dim: usize,
    vocab: usize,
    eps: f32,
    rope_base: f32,
    rope_scale: f32,
    n_rot: usize,
    rope_neox: bool,
    ctx_train: usize,
    /// Sliding-window attention layers (0 = none): window size and RoPE base/scale.
    swa_window: usize = 0,
    rope_base_swa: f32 = 10000,
    rope_scale_swa: f32 = 1,
    key_length_swa: usize = 0,
    n_rot_swa: usize = 0,
    final_logit_softcapping: f32 = 0,
    n_embd_per_layer: usize = 0,
    shared_kv_layers: usize = 0,
    /// Short-convolution layers (LFM): kernel length, 0 = none.
    conv_kernel: usize = 0,
    /// Token embeddings are multiplied by this (Gemma: sqrt(dim)).
    embd_scale: f32 = 1,
    /// Scale of the attention logits q.k.
    attn_scale: f32 = 1,
    /// GELU-gated FFN (Gemma) instead of SiLU.
    gelu: bool = false,
    /// MoE (Mixture of Experts) configuration (0 = dense model).
    n_experts: usize = 0,
    n_experts_used: usize = 0,
    n_shared_experts: usize = 0,
    expert_feed_forward_length: usize = 0,
    expert_weights_scale: f32 = 1.0,
};

pub const Tensor = struct {
    typ: GgmlType,
    data: [*]const u8,
    rows: usize,
    cols: usize,
    /// Bytes per row, or per 8-row group when `x8`.
    row_bytes: usize,
    /// Repacked into 8-row interleaved groups (repack.zig).
    x8: bool = false,

    inline fn row(t: *const Tensor, r: usize) [*]const u8 {
        return t.data + r * t.row_bytes;
    }

    inline fn group(t: *const Tensor, r: usize) [*]const u8 {
        return t.data + (r / 8) * t.row_bytes;
    }

    fn canRepack(t: *const Tensor) bool {
        return repack.supported(t.typ) and t.rows % 8 == 0;
    }

    fn materializedBytes(t: *const Tensor, use_repack: bool) usize {
        const n = if (use_repack and t.canRepack()) t.rows / 8 * repack.groupBytes(t.typ, t.cols) else t.rows * t.row_bytes;
        return std.mem.alignForward(usize, n, 64);
    }
};

pub const Attention = struct {
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    /// Per-head sigmoid output gate (LFM3).
    gate: ?Tensor = null,
    bq: ?[]const f32 = null,
    bk: ?[]const f32 = null,
    bv: ?[]const f32 = null,
    q_norm: ?[]const f32 = null,
    k_norm: ?[]const f32 = null,
    n_kv_heads: usize,
    /// Sliding-window layer: local RoPE base and a `swa_window` mask.
    swa: bool = false,
    /// Index of this layer's KV cache region.
    kv_slot: usize,
    head_dim: usize = 0,
    q_dim: usize = 0,
    kv_dim: usize = 0,
    attn_scale: f32 = 1.0,
    n_rot: usize = 0,
    has_kv: bool = true,
};

/// LFM gated short convolution: y = C * conv(B * x), all from one in_proj.
pub const ShortConv = struct {
    in_proj: Tensor,
    out_proj: Tensor,
    /// Depthwise causal kernel, transposed to [tap][channel].
    kernel: []const f32,
    /// Index of this layer's convolution state.
    slot: usize,
};

pub const DenseFfn = struct {
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
};

pub const Expert = struct {
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
};

pub const MoEFfn = struct {
    router: Tensor,
    experts: []Expert,
    shared: ?DenseFfn = null,
};

pub const Ffn = union(enum) {
    dense: DenseFfn,
    moe: MoEFfn,
};

pub const Layer = struct {
    attn_norm: []const f32,
    ffn_norm: []const f32,
    /// Sandwich norms applied to the mixer / FFN output before the residual add.
    post_mix_norm: ?[]const f32 = null,
    post_ffn_norm: ?[]const f32 = null,
    mixer: union(enum) { attention: Attention, shortconv: ShortConv },
    ffn: Ffn,
    inp_gate: ?Tensor = null,
    proj: ?Tensor = null,
    per_layer_post_norm: ?[]const f32 = null,
    layer_output_scale: ?f32 = null,

    pub inline fn ffnUp(L: *const Layer) *const Tensor {
        return switch (L.ffn) {
            .dense => |*d| &d.w_up,
            .moe => |*m| if (m.experts.len > 0) &m.experts[0].w_up else if (m.shared) |*s| &s.w_up else &m.router,
        };
    }

    pub inline fn ffnDown(L: *const Layer) *const Tensor {
        return switch (L.ffn) {
            .dense => |*d| &d.w_down,
            .moe => |*m| if (m.experts.len > 0) &m.experts[0].w_down else if (m.shared) |*s| &s.w_down else &m.router,
        };
    }
};

pub const LoadOptions = struct {
    /// Interleave weights for the x8 kernels (otherwise plain per-row kernels).
    repack: bool = true,
    /// Fast two-stage coarse-to-fine output projection on lm_head.
    prune_vocab: bool = true,
};

pub const Model = struct {
    file: gguf.File,
    cfg: Config,
    /// Token embeddings, read row-wise straight from the mapped file.
    embd: Tensor,
    output: Tensor,
    tok_embd_per_layer: ?Tensor = null,
    per_layer_model_proj: ?Tensor = null,
    per_layer_proj_norm: ?[]const f32 = null,
    out_norm: []const f32,
    layers: []Layer,
    inv_freq: []f32,
    /// RoPE frequencies of sliding-window layers (empty when there are none).
    inv_freq_swa: []f32 = &.{},
    n_attn_layers: usize = 0,
    n_conv_layers: usize = 0,
    owned: std.ArrayList([]f32) = .empty,
    /// Anonymous huge-page region holding every matmul weight.
    weights: ?[]align(std.heap.page_size_min) u8 = null,
    /// 2-bit copy of the output weights (repack.packCoarse layout) for the
    /// coarse pass of two-stage vocab pruning; `output_coarse_bytes` per group.
    output_coarse: ?[]align(64) u8 = null,
    output_coarse_bytes: usize = 0,

    pub fn load(gpa: Allocator, io: std.Io, path: []const u8, pool: *Pool, opts: LoadOptions) !*Model {
        const self = try gpa.create(Model);
        errdefer gpa.destroy(self);
        self.* = .{
            .file = try gguf.File.open(gpa, io, path),
            .cfg = undefined,
            .embd = undefined,
            .output = undefined,
            .out_norm = &.{},
            .layers = &.{},
            .inv_freq = &.{},
        };
        errdefer self.deinit(gpa);
        try self.readConfig();
        try self.bindTensors(gpa);
        try self.buildRope(gpa);
        try self.materialize(gpa, pool, opts.repack, opts.prune_vocab);
        return self;
    }

    pub fn deinit(self: *Model, gpa: Allocator) void {
        for (self.owned.items) |b| gpa.free(b);
        self.owned.deinit(gpa);
        for (self.layers) |*L| {
            switch (L.ffn) {
                .dense => {},
                .moe => |*m| gpa.free(m.experts),
            }
        }
        gpa.free(self.layers);
        gpa.free(self.inv_freq);
        gpa.free(self.inv_freq_swa);
        if (self.weights) |w| posix.munmap(w);
        if (self.output_coarse) |c| gpa.free(c);
        self.file.deinit(gpa);
    }

    pub fn destroy(self: *Model, gpa: Allocator) void {
        self.deinit(gpa);
        gpa.destroy(self);
    }

    fn matmulTensors(self: *Model, gpa: Allocator) ![]*Tensor {
        var list: std.ArrayList(*Tensor) = .empty;
        errdefer list.deinit(gpa);
        for (self.layers) |*L| {
            switch (L.mixer) {
                .attention => |*A| {
                    try list.appendSlice(gpa, &.{ &A.wq, &A.wk, &A.wv, &A.wo });
                    if (A.gate) |*g| try list.append(gpa, g);
                },
                .shortconv => |*C| try list.appendSlice(gpa, &.{ &C.in_proj, &C.out_proj }),
            }
            switch (L.ffn) {
                .dense => |*d| try list.appendSlice(gpa, &.{ &d.w_gate, &d.w_up, &d.w_down }),
                .moe => |*m| {
                    try list.append(gpa, &m.router);
                    for (m.experts) |*exp| {
                        try list.appendSlice(gpa, &.{ &exp.w_gate, &exp.w_up, &exp.w_down });
                    }
                    if (m.shared) |*sh| {
                        try list.appendSlice(gpa, &.{ &sh.w_gate, &sh.w_up, &sh.w_down });
                    }
                },
            }
            if (L.inp_gate) |*g| try list.append(gpa, g);
            if (L.proj) |*p| try list.append(gpa, p);
        }
        if (self.per_layer_model_proj) |*p| try list.append(gpa, p);
        try list.append(gpa, &self.output);
        return list.toOwnedSlice(gpa);
    }

    const RepackJob = struct { src: *const Tensor, dst: [*]u8, gbytes: usize, chunks: Chunks };

    fn repackWorker(j: *RepackJob, _: usize, _: usize) void {
        const t = j.src;
        const nb = t.cols / t.typ.layout().?.block;
        while (j.chunks.grab()) |range| for (range[0]..range[1]) |g| {
            var rows: [8][*]const u8 = undefined;
            for (&rows, 0..) |*p, i| p.* = t.row(g * 8 + i);
            repack.repackGroup(t.typ, rows, nb, j.dst + g * j.gbytes);
        };
    }

    const CoarseBuildJob = struct { src: *const Tensor, dst: [*]u8, gbytes: usize, scratch: []f32, chunks: Chunks };

    fn coarseBuildWorker(j: *CoarseBuildJob, tid: usize, _: usize) void {
        const t = j.src;
        const buf = j.scratch[tid * 8 * t.cols ..][0 .. 8 * t.cols];
        while (j.chunks.grab()) |range| for (range[0]..range[1]) |g| {
            var rows: [8][]const f32 = undefined;
            for (&rows, 0..) |*r, i| {
                const row = buf[i * t.cols ..][0..t.cols];
                quant.dequantRow(t.typ, t.row(g * 8 + i), row);
                r.* = row;
            }
            repack.packCoarse(rows, j.dst + g * j.gbytes);
        };
    }

    /// Move every matmul weight into one huge-page region, repacking when possible.
    fn materialize(self: *Model, gpa: Allocator, pool: *Pool, use_repack: bool, prune_vocab: bool) !void {
        const ts = try self.matmulTensors(gpa);
        defer gpa.free(ts);
        var total: usize = 0;
        for (ts) |t| total += t.materializedBytes(use_repack);
        const len = std.mem.alignForward(usize, total, 2 << 20);
        const region = try posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        self.weights = region;
        posix.madvise(region.ptr, len, posix.MADV.HUGEPAGE) catch {};
        // The 2-bit coarse copy of lm_head is quantized from the GGUF rows,
        // before the loop below repacks the output weights.
        if (prune_vocab and use_repack and self.output.canRepack() and self.output.cols % 32 == 0) {
            const t = &self.output;
            const gbytes = repack.coarseGroupBytes(t.cols);
            const coarse = try gpa.alignedAlloc(u8, .@"64", t.rows / 8 * gbytes);
            self.output_coarse = coarse;
            self.output_coarse_bytes = gbytes;
            const scratch = try gpa.alloc(f32, pool.nth * 8 * t.cols);
            defer gpa.free(scratch);
            var job: CoarseBuildJob = .{ .src = t, .dst = coarse.ptr, .gbytes = gbytes, .scratch = scratch, .chunks = .init(t.rows / 8, 16) };
            pool.run(&job, coarseBuildWorker);
        }
        var off: usize = 0;
        for (ts) |t| {
            const dst = region.ptr + off;
            off += t.materializedBytes(use_repack);
            if (use_repack and t.canRepack()) {
                const gbytes = repack.groupBytes(t.typ, t.cols);
                var job: RepackJob = .{ .src = t, .dst = dst, .gbytes = gbytes, .chunks = .init(t.rows / 8, 16) };
                pool.run(&job, repackWorker);
                t.data = dst;
                t.row_bytes = gbytes;
                t.x8 = true;
            } else {
                @memcpy(dst[0 .. t.rows * t.row_bytes], t.data[0 .. t.rows * t.row_bytes]);
                t.data = dst;
            }
        }
    }

    fn key(self: *const Model, buf: []u8, comptime suffix: []const u8) []const u8 {
        return std.fmt.bufPrint(buf, "{s}." ++ suffix, .{@tagName(self.cfg.arch)}) catch unreachable;
    }

    fn readConfig(self: *Model) !void {
        const f = &self.file;
        const arch_s = f.getString("general.architecture") orelse return error.MissingArchitecture;
        const arch = std.meta.stringToEnum(Arch, arch_s) orelse if (std.mem.eql(u8, arch_s, "qwen2_moe"))
            Arch.qwen2moe
        else if (std.mem.eql(u8, arch_s, "gemma-4") or std.mem.eql(u8, arch_s, "gemma4"))
            Arch.gemma4
        else {
            std.log.err("unsupported architecture '{s}' (supported: llama, qwen2, qwen3, lfm2, lfm3, gemma3, gemma4, qwen2moe, mixtral, deepseek2)", .{arch_s});
            return error.UnsupportedArchitecture;
        };
        self.cfg.arch = arch;
        var kb: [128]u8 = undefined;
        const req = struct {
            fn u(file: *const gguf.File, k: []const u8) !usize {
                return @intCast(file.getUint(k) orelse {
                    std.log.err("missing metadata key {s}", .{k});
                    return error.MissingMetadata;
                });
            }
        }.u;
        const c = &self.cfg;
        c.dim = try req(f, self.key(&kb, "embedding_length"));
        c.n_layers = try req(f, self.key(&kb, "block_count"));
        c.n_heads = try req(f, self.key(&kb, "attention.head_count"));
        c.n_kv_heads = 0;
        for (0..c.n_layers) |i| c.n_kv_heads = @max(c.n_kv_heads, self.layerKvHeads(i));
        if (c.n_kv_heads == 0) return error.InvalidConfig;
        c.ffn_dim = blk: {
            const k = self.key(&kb, "feed_forward_length");
            if (f.getArray(k)) |a| {
                var max_ffn: usize = 0;
                for (0..a.len) |idx| {
                    const val: usize = switch (a.elem) {
                        .i32 => @intCast(@max(0, a.item(i32, idx))),
                        .u32 => a.item(u32, idx),
                        .u64 => a.item(u64, idx),
                        .i64 => @intCast(@max(0, a.item(i64, idx))),
                        else => 0,
                    };
                    max_ffn = @max(max_ffn, val);
                }
                break :blk max_ffn;
            }
            break :blk try req(f, k);
        };
        c.head_dim = @intCast(f.getUint(self.key(&kb, "attention.key_length")) orelse c.dim / c.n_heads);
        c.key_length_swa = @intCast(f.getUint(self.key(&kb, "attention.key_length_swa")) orelse c.head_dim);
        c.q_dim = c.n_heads * c.head_dim;
        c.kv_dim = c.n_kv_heads * c.head_dim;
        c.eps = @floatCast(f.getFloat(self.key(&kb, "attention.layer_norm_rms_epsilon")) orelse 1e-5);
        c.rope_base = @floatCast(f.getFloat(self.key(&kb, "rope.freq_base")) orelse 10000);
        c.n_rot = @intCast(f.getUint(self.key(&kb, "rope.dimension_count")) orelse c.head_dim);
        c.n_rot_swa = @intCast(f.getUint(self.key(&kb, "rope.dimension_count_swa")) orelse c.key_length_swa);
        c.rope_neox = arch != .llama;
        c.ctx_train = @intCast(f.getUint(self.key(&kb, "context_length")) orelse 4096);
        c.rope_scale = 1;
        if (f.getString(self.key(&kb, "rope.scaling.type"))) |t| {
            if (std.mem.eql(u8, t, "linear")) {
                c.rope_scale = @floatCast(1.0 / (f.getFloat(self.key(&kb, "rope.scaling.factor")) orelse 1));
            } else if (!std.mem.eql(u8, t, "none")) {
                std.log.warn("rope scaling '{s}' not implemented; long contexts may degrade", .{t});
            }
        }
        c.swa_window = @intCast(f.getUint(self.key(&kb, "attention.sliding_window")) orelse 0);
        // Gemma 3/4 local layers default to base 10000 without scaling; LFM2's
        // windowed layers use the global RoPE.
        c.rope_base_swa = @floatCast(f.getFloat(self.key(&kb, "rope.freq_base_swa")) orelse if (arch == .gemma3 or arch == .gemma4) 10000 else c.rope_base);
        c.rope_scale_swa = if (arch == .lfm2) c.rope_scale else 1;
        c.conv_kernel = @intCast(f.getUint(self.key(&kb, "shortconv.l_cache")) orelse 0);
        c.attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
        c.embd_scale = if (arch == .gemma3 or arch == .gemma4) @sqrt(@as(f32, @floatFromInt(c.dim))) else 1;
        c.gelu = (arch == .gemma3 or arch == .gemma4);
        c.final_logit_softcapping = @floatCast(f.getFloat(self.key(&kb, "final_logit_softcapping")) orelse 0);
        c.n_embd_per_layer = @intCast(f.getUint(self.key(&kb, "embedding_length_per_layer_input")) orelse 0);
        c.shared_kv_layers = @intCast(f.getUint(self.key(&kb, "attention.shared_kv_layers")) orelse 0);
        if (arch == .gemma4) {
            c.n_rot = 128;
            c.n_rot_swa = 256;
        }
        c.n_experts = @intCast(f.getUint(self.key(&kb, "expert_count")) orelse f.getUint("expert_count") orelse 0);
        c.n_experts_used = @intCast(f.getUint(self.key(&kb, "expert_used_count")) orelse f.getUint("expert_used_count") orelse @min(c.n_experts, 2));
        c.n_shared_experts = @intCast(f.getUint(self.key(&kb, "expert_shared_count")) orelse f.getUint("expert_shared_count") orelse 0);
        c.expert_feed_forward_length = @intCast(f.getUint(self.key(&kb, "expert_feed_forward_length")) orelse c.ffn_dim);
        c.expert_weights_scale = @floatCast(f.getFloat(self.key(&kb, "expert_weights_scale")) orelse 1.0);
        if (arch == .gemma3) {
            // gemma_pytorch: the 27B scales queries by 1 / sqrt(dim / n_heads).
            if (c.n_layers == 62) c.attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.dim / c.n_heads)));
        }
        for (0..c.n_layers) |i| {
            const kv = self.layerKvHeads(i);
            if (kv == 0 and c.conv_kernel < 2) return error.InvalidConfig;
            if (kv != 0 and c.n_heads % kv != 0) return error.InvalidConfig;
        }
        if (c.n_rot > c.head_dim or c.n_rot % 2 != 0) return error.InvalidConfig;
    }

    /// KV heads of layer `i`; 0 marks a short-convolution layer. The GGUF
    /// value is either one integer or a per-layer array.
    fn layerKvHeads(self: *const Model, i: usize) usize {
        var kb: [128]u8 = undefined;
        const k = self.key(&kb, "attention.head_count_kv");
        if (self.file.getArray(k)) |a| {
            if (i >= a.len) return 0;
            return switch (a.elem) {
                .i32 => @intCast(@max(0, a.item(i32, i))),
                .u32 => a.item(u32, i),
                .u16 => a.item(u16, i),
                .u8 => a.item(u8, i),
                else => 0,
            };
        }
        return @intCast(self.file.getUint(k) orelse (self.file.getUint(self.key(&kb, "attention.head_count")) orelse 0));
    }

    fn layerIsSwa(self: *const Model, i: usize) bool {
        if (self.cfg.swa_window == 0) return false;
        var kb: [128]u8 = undefined;
        const k = self.key(&kb, "attention.sliding_window_pattern");
        if (self.file.getArray(k)) |a| return a.elem == .bool and i < a.len and a.item(u8, i) != 0;
        return switch (self.cfg.arch) {
            // Every attention layer is windowed.
            .lfm2 => true,
            // n - 1 local layers, then a global one (period 6 unless given).
            .gemma3 => blk: {
                const n = self.file.getUint(k) orelse 6;
                break :blk n == 0 or i % n < n - 1;
            },
            .gemma4 => blk: {
                if (self.file.getArray(k)) |a| break :blk a.elem == .bool and i < a.len and a.item(u8, i) != 0;
                break :blk false;
            },
            else => false,
        };
    }

    fn layerFfnDim(self: *const Model, i: usize) usize {
        var kb: [128]u8 = undefined;
        const k = self.key(&kb, "feed_forward_length");
        if (self.file.getArray(k)) |a| {
            if (i >= a.len) return self.cfg.ffn_dim;
            return switch (a.elem) {
                .i32 => @intCast(@max(0, a.item(i32, i))),
                .u32 => a.item(u32, i),
                .u64 => a.item(u64, i),
                .i64 => @intCast(@max(0, a.item(i64, i))),
                else => self.cfg.ffn_dim,
            };
        }
        return self.cfg.ffn_dim;
    }

    fn optScalar(self: *Model, name: []const u8) ?f32 {
        const ti = self.file.tensors.get(name) orelse return null;
        if (ti.typ != .f32 or ti.nelems() != 1 or ti.data.len < 4) return null;
        return @bitCast(std.mem.readInt(u32, ti.data[0..4], .little));
    }

    fn tensor(self: *Model, name: []const u8, cols: usize, rows: ?usize) !Tensor {
        const ti = self.file.tensors.get(name) orelse {
            std.log.err("missing tensor {s}", .{name});
            return error.MissingTensor;
        };
        if (!quant.isSupported(ti.typ)) {
            std.log.err("tensor {s}: unsupported type {d} (supported: F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K Q5_K Q6_K)", .{ name, @backingInt(ti.typ) });
            return error.UnsupportedTensorType;
        }
        if (ti.dims[0] != cols or (rows != null and ti.dims[1] != rows.?) or ti.n_dims > 2) {
            std.log.err("tensor {s}: unexpected shape {any}", .{ name, ti.dims[0..ti.n_dims] });
            return error.BadTensorShape;
        }
        if (cols % 256 != 0 and quant.actKind(ti.typ) == .q8k) return error.BadTensorShape;
        return .{
            .typ = ti.typ,
            .data = ti.data.ptr,
            .rows = @intCast(ti.dims[1]),
            .cols = cols,
            .row_bytes = ti.typ.rowBytes(cols).?,
        };
    }

    fn vector(self: *Model, gpa: Allocator, name: []const u8, n: usize) ![]const f32 {
        const t = try self.tensor(name, n, 1);
        if (t.typ == .f32) return @as([*]const f32, @ptrCast(@alignCast(t.data)))[0..n];
        const buf = try gpa.alloc(f32, n);
        errdefer gpa.free(buf);
        quant.dequantRow(t.typ, t.data, buf);
        try self.owned.append(gpa, buf);
        return buf;
    }

    fn optVector(self: *Model, gpa: Allocator, name: []const u8, n: usize) !?[]const f32 {
        if (!self.file.tensors.contains(name)) return null;
        return try self.vector(gpa, name, n);
    }

    fn bindTensors(self: *Model, gpa: Allocator) !void {
        const c = &self.cfg;
        self.embd = try self.tensor("token_embd.weight", c.dim, null);
        c.vocab = self.embd.rows;
        self.output = if (self.file.tensors.contains("output.weight"))
            try self.tensor("output.weight", c.dim, c.vocab)
        else
            self.embd;
        // LFM2 names its final norm token_embd_norm.
        self.out_norm = try self.vector(gpa, if (c.arch == .lfm2) "token_embd_norm.weight" else "output_norm.weight", c.dim);
        if (c.n_embd_per_layer > 0) {
            const total_ple_dim = c.n_embd_per_layer * c.n_layers;
            self.tok_embd_per_layer = try self.tensor("per_layer_token_embd.weight", total_ple_dim, c.vocab);
            self.per_layer_model_proj = try self.tensor("per_layer_model_proj.weight", c.dim, total_ple_dim);
            self.per_layer_proj_norm = try self.vector(gpa, "per_layer_proj_norm.weight", c.n_embd_per_layer);
        }
        self.layers = try gpa.alloc(Layer, c.n_layers);
        if (c.n_layers > max_layers) return error.InvalidConfig;
        var nb: [256]u8 = undefined;
        var kv_source: [max_layers]usize = undefined;
        for (0..c.n_layers) |i| kv_source[i] = i;
        if (c.shared_kv_layers > 0 and c.n_layers > c.shared_kv_layers) {
            const first_shared = c.n_layers - c.shared_kv_layers;
            for (first_shared..c.n_layers) |i| {
                const is_swa = self.layerIsSwa(i);
                var src: usize = i;
                var j: usize = i;
                while (j > 0) {
                    j -= 1;
                    if (j < first_shared and self.layerIsSwa(j) == is_swa) {
                        src = j;
                        break;
                    }
                }
                kv_source[i] = src;
            }
        }
        var layer_kv_slots: [max_layers]usize = undefined;
        for (self.layers, 0..) |*L, i| {
            const n = struct {
                fn f(buf: []u8, l: usize, comptime s: []const u8) []const u8 {
                    return std.fmt.bufPrint(buf, "blk.{d}." ++ s, .{l}) catch unreachable;
                }
            }.f;
            var layer_ffn: Ffn = undefined;
            if (c.n_experts > 0) {
                var experts = try gpa.alloc(Expert, c.n_experts);
                const gate_exps_name = n(&nb, i, "ffn_gate_exps.weight");
                if (self.file.tensors.contains(gate_exps_name)) {
                    const g_info = self.file.tensors.get(gate_exps_name).?;
                    const u_info = self.file.tensors.get(n(&nb, i, "ffn_up_exps.weight")).?;
                    const d_info = self.file.tensors.get(n(&nb, i, "ffn_down_exps.weight")).?;
                    const g_row_bytes = g_info.typ.rowBytes(c.dim) orelse return error.UnsupportedQuantType;
                    const u_row_bytes = u_info.typ.rowBytes(c.dim) orelse return error.UnsupportedQuantType;
                    const d_row_bytes = d_info.typ.rowBytes(c.expert_feed_forward_length) orelse return error.UnsupportedQuantType;
                    const g_slice_bytes = c.expert_feed_forward_length * g_row_bytes;
                    const u_slice_bytes = c.expert_feed_forward_length * u_row_bytes;
                    const d_slice_bytes = c.dim * d_row_bytes;

                    for (0..c.n_experts) |e| {
                        experts[e] = .{
                            .w_gate = .{
                                .typ = g_info.typ,
                                .data = g_info.data.ptr + e * g_slice_bytes,
                                .rows = c.expert_feed_forward_length,
                                .cols = c.dim,
                                .row_bytes = g_row_bytes,
                            },
                            .w_up = .{
                                .typ = u_info.typ,
                                .data = u_info.data.ptr + e * u_slice_bytes,
                                .rows = c.expert_feed_forward_length,
                                .cols = c.dim,
                                .row_bytes = u_row_bytes,
                            },
                            .w_down = .{
                                .typ = d_info.typ,
                                .data = d_info.data.ptr + e * d_slice_bytes,
                                .rows = c.dim,
                                .cols = c.expert_feed_forward_length,
                                .row_bytes = d_row_bytes,
                            },
                        };
                    }
                } else {
                    for (0..c.n_experts) |e| {
                        var eb: [96]u8 = undefined;
                        const ne = struct {
                            fn f(buf: []u8, l: usize, comptime s: []const u8, exp_idx: usize) []const u8 {
                                return std.fmt.bufPrint(buf, "blk.{d}." ++ s ++ ".{d}.weight", .{ l, exp_idx }) catch unreachable;
                            }
                        }.f;
                        experts[e] = .{
                            .w_gate = try self.tensor(ne(&eb, i, "ffn_gate", e), c.dim, c.expert_feed_forward_length),
                            .w_up = try self.tensor(ne(&eb, i, "ffn_up", e), c.dim, c.expert_feed_forward_length),
                            .w_down = try self.tensor(ne(&eb, i, "ffn_down", e), c.expert_feed_forward_length, c.dim),
                        };
                    }
                }
                const router = try self.tensor(n(&nb, i, "ffn_gate_inp.weight"), c.dim, c.n_experts);
                const shared_gate_name = n(&nb, i, "ffn_gate_shexp.weight");
                const has_shexp = self.file.tensors.contains(shared_gate_name) or (c.n_shared_experts > 0 and self.file.tensors.contains(n(&nb, i, "ffn_gate.weight")));
                const shared: ?DenseFfn = if (has_shexp) blk: {
                    const sg_name = if (self.file.tensors.contains(shared_gate_name)) shared_gate_name else n(&nb, i, "ffn_gate.weight");
                    const su_name = if (self.file.tensors.contains(n(&nb, i, "ffn_up_shexp.weight"))) n(&nb, i, "ffn_up_shexp.weight") else n(&nb, i, "ffn_up.weight");
                    const sd_name = if (self.file.tensors.contains(n(&nb, i, "ffn_down_shexp.weight"))) n(&nb, i, "ffn_down_shexp.weight") else n(&nb, i, "ffn_down.weight");
                    break :blk DenseFfn{
                        .w_gate = try self.tensor(sg_name, c.dim, c.ffn_dim),
                        .w_up = try self.tensor(su_name, c.dim, c.ffn_dim),
                        .w_down = try self.tensor(sd_name, c.ffn_dim, c.dim),
                    };
                } else null;

                layer_ffn = .{ .moe = .{
                    .router = router,
                    .experts = experts,
                    .shared = shared,
                } };
            } else {
                const l_ffn_dim = self.layerFfnDim(i);
                layer_ffn = .{ .dense = .{
                    .w_gate = try self.tensor(n(&nb, i, "ffn_gate.weight"), c.dim, l_ffn_dim),
                    .w_up = try self.tensor(n(&nb, i, "ffn_up.weight"), c.dim, l_ffn_dim),
                    .w_down = try self.tensor(n(&nb, i, "ffn_down.weight"), l_ffn_dim, c.dim),
                } };
            }
            L.* = .{
                .attn_norm = try self.vector(gpa, n(&nb, i, "attn_norm.weight"), c.dim),
                .ffn_norm = try self.vector(gpa, n(&nb, i, "ffn_norm.weight"), c.dim),
                .post_mix_norm = try self.optVector(gpa, n(&nb, i, "post_attention_norm.weight"), c.dim),
                .post_ffn_norm = try self.optVector(gpa, n(&nb, i, "post_ffw_norm.weight"), c.dim),
                .mixer = undefined,
                .ffn = layer_ffn,
            };
            const kv = self.layerKvHeads(i);
            if (kv == 0) {
                L.mixer = .{ .shortconv = .{
                    .in_proj = try self.tensor(n(&nb, i, "shortconv.in_proj.weight"), c.dim, 3 * c.dim),
                    .out_proj = try self.tensor(n(&nb, i, "shortconv.out_proj.weight"), c.dim, c.dim),
                    .kernel = try self.convKernel(gpa, n(&nb, i, "shortconv.conv.weight")),
                    .slot = self.n_conv_layers,
                } };
                self.n_conv_layers += 1;
            } else {
                const is_swa = self.layerIsSwa(i);
                const is_shared = (c.shared_kv_layers > 0 and i >= c.n_layers - c.shared_kv_layers);
                const has_kv = !is_shared;
                const kv_slot = if (has_kv) self.n_attn_layers else layer_kv_slots[kv_source[i]];
                if (has_kv) {
                    layer_kv_slots[i] = kv_slot;
                    self.n_attn_layers += 1;
                }
                const l_hd = if (c.arch == .gemma4 and is_swa) c.key_length_swa else c.head_dim;
                const l_q_dim = c.n_heads * l_hd;
                const l_kv_dim = kv * l_hd;
                const l_n_rot = if (c.arch == .gemma4 and is_swa) c.n_rot_swa else c.n_rot;
                const l_attn_scale: f32 = if (c.arch == .gemma4) 1.0 else 1.0 / @sqrt(@as(f32, @floatFromInt(l_hd)));

                L.mixer = .{ .attention = .{
                    .wq = try self.tensor(n(&nb, i, "attn_q.weight"), c.dim, l_q_dim),
                    .wk = try self.tensor(n(&nb, i, "attn_k.weight"), c.dim, l_kv_dim),
                    .wv = try self.tensor(n(&nb, i, "attn_v.weight"), c.dim, l_kv_dim),
                    .wo = try self.tensor(n(&nb, i, "attn_output.weight"), l_q_dim, c.dim),
                    .gate = if (self.file.tensors.contains(n(&nb, i, "attn_gate.weight"))) try self.tensor(n(&nb, i, "attn_gate.weight"), c.dim, c.n_heads) else null,
                    .bq = try self.optVector(gpa, n(&nb, i, "attn_q.bias"), l_q_dim),
                    .bk = try self.optVector(gpa, n(&nb, i, "attn_k.bias"), l_kv_dim),
                    .bv = try self.optVector(gpa, n(&nb, i, "attn_v.bias"), l_kv_dim),
                    .q_norm = try self.optVector(gpa, n(&nb, i, "attn_q_norm.weight"), l_hd),
                    .k_norm = try self.optVector(gpa, n(&nb, i, "attn_k_norm.weight"), l_hd),
                    .n_kv_heads = kv,
                    .swa = is_swa,
                    .kv_slot = kv_slot,
                    .head_dim = l_hd,
                    .q_dim = l_q_dim,
                    .kv_dim = l_kv_dim,
                    .attn_scale = l_attn_scale,
                    .n_rot = l_n_rot,
                    .has_kv = has_kv,
                } };
            }
            if (c.n_embd_per_layer > 0) {
                L.inp_gate = if (self.file.tensors.contains(n(&nb, i, "inp_gate.weight"))) try self.tensor(n(&nb, i, "inp_gate.weight"), c.dim, c.n_embd_per_layer) else null;
                L.proj = if (self.file.tensors.contains(n(&nb, i, "proj.weight"))) try self.tensor(n(&nb, i, "proj.weight"), c.n_embd_per_layer, c.dim) else null;
                L.per_layer_post_norm = try self.optVector(gpa, n(&nb, i, "post_norm.weight"), c.dim);
                L.layer_output_scale = self.optScalar(n(&nb, i, "layer_output_scale.weight"));
            }
        }
    }

    /// Short-conv taps, stored [channel][tap] in GGUF, transposed to [tap][channel].
    fn convKernel(self: *Model, gpa: Allocator, name: []const u8) ![]const f32 {
        const c = &self.cfg;
        var row: [64]f32 = undefined;
        if (c.conv_kernel > row.len) return error.InvalidConfig;
        const t = try self.tensor(name, c.conv_kernel, c.dim);
        const out = try gpa.alloc(f32, c.conv_kernel * c.dim);
        errdefer gpa.free(out);
        try self.owned.append(gpa, out);
        for (0..c.dim) |ch| {
            quant.dequantRow(t.typ, t.row(ch), row[0..c.conv_kernel]);
            for (0..c.conv_kernel) |k| out[k * c.dim + ch] = row[k];
        }
        return out;
    }

    fn ropeFreqs(base: f32, scale: f32, factors: ?[]const f32, out: []f32) void {
        rope.ropeFreqs(base, scale, factors, out);
    }

    fn buildRope(self: *Model, gpa: Allocator) !void {
        const c = &self.cfg;
        const half = c.n_rot / 2;
        self.inv_freq = try gpa.alloc(f32, half);
        // rope_freqs factors belong to the full-attention layers only.
        rope.ropeFreqs(c.rope_base, c.rope_scale, if (c.arch == .gemma4) null else try self.optVector(gpa, "rope_freqs.weight", half), self.inv_freq);
        if (c.swa_window > 0) {
            const half_swa = if (c.arch == .gemma4) c.n_rot_swa / 2 else half;
            self.inv_freq_swa = try gpa.alloc(f32, half_swa);
            rope.ropeFreqs(c.rope_base_swa, c.rope_scale_swa, null, self.inv_freq_swa);
        }
    }
};

// ---------------------------------------------------------------------------
// Runtime state

/// Activation rows plus lazily-built quantized copies for the matmul kernels.
const Act = struct {
    n: usize,
    ntok: usize = 0,
    f: []f32,
    q8k: []quant.Q8K,
    q8b: []quant.Q8B,
    have_q8k: bool = false,
    have_q8b: bool = false,

    fn init(gpa: Allocator, n: usize, bmax: usize) !Act {
        const f = try gpa.alignedAlloc(f32, .@"64", n * bmax);
        errdefer gpa.free(f);
        const q8k = try gpa.alloc(quant.Q8K, if (n % 256 == 0) n / 256 * bmax else 0);
        errdefer gpa.free(q8k);
        return .{ .n = n, .f = f, .q8k = q8k, .q8b = try gpa.alloc(quant.Q8B, n / 32 * bmax) };
    }

    fn deinit(a: *Act, gpa: Allocator) void {
        gpa.free(a.f);
        gpa.free(a.q8k);
        gpa.free(a.q8b);
    }

    fn reset(a: *Act, ntok: usize) void {
        a.ntok = ntok;
        a.have_q8k = false;
        a.have_q8b = false;
    }

    fn resetDim(a: *Act, n: usize, ntok: usize) void {
        a.n = n;
        a.ntok = ntok;
        a.have_q8k = false;
        a.have_q8b = false;
    }

    fn quantRows(a: *Act, kind: quant.ActKind, t0: usize, t1: usize) void {
        for (t0..t1) |t| {
            const src = a.f[t * a.n ..][0..a.n];
            switch (kind) {
                .q8k => quant.quantizeQ8K(src, a.q8k[t * (a.n / 256) ..][0 .. a.n / 256]),
                .q8b => quant.quantizeQ8B(src, a.q8b[t * (a.n / 32) ..][0 .. a.n / 32]),
                .f32 => {},
            }
        }
    }

    const QuantJob = struct { a: *Act, kind: quant.ActKind, chunks: Chunks };

    fn quantWorker(j: *QuantJob, _: usize, _: usize) void {
        while (j.chunks.grab()) |r| j.a.quantRows(j.kind, r[0], r[1]);
    }

    fn prepare(a: *Act, kind: quant.ActKind, pool: *Pool) void {
        switch (kind) {
            .f32 => return,
            .q8k => if (a.have_q8k) return else {
                a.have_q8k = true;
            },
            .q8b => if (a.have_q8b) return else {
                a.have_q8b = true;
            },
        }
        if (a.ntok * a.n < 32768) return a.quantRows(kind, 0, a.ntok);
        var job: QuantJob = .{ .a = a, .kind = kind, .chunks = .init(a.ntok, 1) };
        pool.run(&job, quantWorker);
    }

    inline fn rowPtr(a: *const Act, comptime kind: quant.ActKind, t: usize) [*]const u8 {
        return switch (kind) {
            .f32 => @ptrCast(a.f.ptr + t * a.n),
            .q8k => @ptrCast(a.q8k.ptr + t * (a.n / 256)),
            .q8b => @ptrCast(a.q8b.ptr + t * (a.n / 32)),
        };
    }
};

const MatOut = struct {
    w: *const Tensor,
    out: [*]f32,
    /// Distance between consecutive tokens in `out`.
    stride: usize,
    accumulate: bool = false,
};

const tok_tile = 8;
const max_rows_chunk = 64;
/// Tokens per kernel call: each weight block is unpacked once per group.
/// 8 tokens keep 16 accumulators live: +6% pp with the 32 registers of
/// AVX-512VL (Zen 5), but they spill on plain AVX2's 16 (-13% on Skylake).
const tok_group = if (quant.has_avx512vl) 8 else 4;

inline fn storeOut(comptime acc: bool, o: *f32, v: f32) void {
    if (acc) o.* += v else o.* = v;
}

/// out[(t-t0)*ostride + (r-r0)] (+)= W[r] . act[t]
fn rowsDot(comptime t: GgmlType, comptime acc: bool, w: *const Tensor, r0: usize, r1: usize, act: *const Act, t0: usize, t1: usize, out: [*]f32, ostride: usize) void {
    const kind = comptime quant.actKind(t);
    var ts = t0;
    while (ts < t1) : (ts += tok_tile) {
        const te = @min(ts + tok_tile, t1);
        for (r0..r1) |r| {
            const row = w.row(r);
            var tk = ts;
            while (tk < te) {
                switch (@min(te - tk, tok_group)) {
                    inline 1...tok_group => |g| {
                        var ptrs: [g][*]const u8 = undefined;
                        inline for (0..g) |k| ptrs[k] = act.rowPtr(kind, tk + k);
                        const v = quant.dotN(t, g, row, ptrs, w.cols);
                        inline for (0..g) |k| storeOut(acc, &out[(tk + k - t0) * ostride + (r - r0)], v[k]);
                        tk += g;
                    },
                    else => unreachable,
                }
            }
        }
    }
}

/// Same contract as rowsDot for an x8 tensor; r0 and r1 are multiples of 8.
fn groupsDot(comptime t: GgmlType, comptime acc: bool, w: *const Tensor, r0: usize, r1: usize, act: *const Act, t0: usize, t1: usize, out: [*]f32, ostride: usize) void {
    const kind = comptime quant.actKind(t);
    if (t1 == t0 + 1) {
        const act_ptr = act.rowPtr(kind, t0);
        const acts = [_][*]const u8{act_ptr};
        var r = r0;
        while (r < r1) : (r += 8) {
            const g = w.group(r);
            const v = repack.gemm(t, 1, g, acts, w.cols)[0];
            const o: *[8]f32 = out[r - r0 ..][0..8];
            o.* = if (acc) @as(quant.V8f, o.*) + v else v;
        }
        return;
    }
    var r = r0;
    while (r < r1) : (r += 8) {
        const g = w.group(r);
        var tk = t0;
        while (tk < t1) {
            switch (@min(t1 - tk, tok_group)) {
                inline 1...tok_group => |n| {
                    var ptrs: [n][*]const u8 = undefined;
                    inline for (0..n) |k| ptrs[k] = act.rowPtr(kind, tk + k);
                    const v = repack.gemm(t, n, g, ptrs, w.cols);
                    inline for (0..n) |k| {
                        const o: *[8]f32 = out[(tk + k - t0) * ostride + (r - r0) ..][0..8];
                        o.* = if (acc) @as(quant.V8f, o.*) + v[k] else v[k];
                    }
                    tk += n;
                },
                else => unreachable,
            }
        }
    }
}

fn rowsDispatch(acc: bool, w: *const Tensor, r0: usize, r1: usize, act: *const Act, t0: usize, t1: usize, out: [*]f32, ostride: usize) void {
    if (w.x8) {
        switch (w.typ) {
            inline .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q4_k, .q5_k, .q6_k => |t| {
                if (acc) groupsDot(t, true, w, r0, r1, act, t0, t1, out, ostride) else groupsDot(t, false, w, r0, r1, act, t0, t1, out, ostride);
            },
            else => unreachable,
        }
        return;
    }
    switch (w.typ) {
        inline .f32, .f16, .bf16, .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q4_k, .q5_k, .q6_k => |t| {
            if (acc) rowsDot(t, true, w, r0, r1, act, t0, t1, out, ostride) else rowsDot(t, false, w, r0, r1, act, t0, t1, out, ostride);
        },
        else => unreachable,
    }
}

const max_outs = 4;

const MatJob = struct {
    outs: []const MatOut,
    act: *const Act,
    ntok: usize,
    tok_offset: usize = 0,
    rows_chunk: usize,
    /// First chunk index of each output (prefix sums), len = outs.len + 1.
    first: [max_outs + 1]usize,
    chunks: Chunks,
};

fn matWorker(job: *MatJob, _: usize, _: usize) void {
    while (job.chunks.grab()) |range| {
        for (range[0]..range[1]) |ci| {
            var m: usize = 0;
            while (ci >= job.first[m + 1]) m += 1;
            const o = &job.outs[m];
            const r0 = (ci - job.first[m]) * job.rows_chunk;
            const r1 = @min(r0 + job.rows_chunk, o.w.rows);
            rowsDispatch(o.accumulate, o.w, r0, r1, job.act, job.tok_offset, job.tok_offset + job.ntok, o.out + r0, o.stride);
        }
    }
}

const FfnJob = struct {
    gate: *const Tensor,
    up: *const Tensor,
    act: *const Act,
    out: []f32,
    ntok: usize,
    tok_offset: usize = 0,
    rows_chunk: usize,
    gelu: bool,
    chunks: Chunks,
};

fn ffnWorkerFusedX8(comptime t: GgmlType, job: *FfnJob) void {
    const kind = comptime quant.actKind(t);
    const ffn_rows = job.gate.rows;
    const gelu = job.gelu;
    while (job.chunks.grab()) |range| {
        const r0 = range[0] * job.rows_chunk;
        const r1 = @min(range[1] * job.rows_chunk, ffn_rows);
        var r = r0;
        while (r < r1) : (r += 8) {
            const gg = job.gate.group(r);
            const gu = job.up.group(r);
            var tk: usize = 0;
            while (tk < job.ntok) {
                switch (@min(job.ntok - tk, tok_group)) {
                    inline 1...tok_group => |n| {
                        var ptrs: [n][*]const u8 = undefined;
                        inline for (0..n) |k| ptrs[k] = job.act.rowPtr(kind, job.tok_offset + tk + k);
                        const vg = repack.gemm(t, n, gg, ptrs, job.gate.cols);
                        const dead_thresh: f32 = if (gelu) -4.0 else -6.0;
                        var all_dead = true;
                        inline for (0..n) |k| {
                            if (@reduce(.Max, vg[k]) > dead_thresh) all_dead = false;
                        }
                        if (all_dead) {
                            inline for (0..n) |k| {
                                const o: *[8]f32 = job.out[(tk + k) * ffn_rows + r ..][0..8];
                                o.* = @splat(0);
                            }
                            tk += n;
                            continue;
                        }
                        const vu = repack.gemm(t, n, gu, ptrs, job.up.cols);
                        inline for (0..n) |k| {
                            const o: *[8]f32 = job.out[(tk + k) * ffn_rows + r ..][0..8];
                            o.* = if (gelu) geluMulV(vg[k], vu[k]) else siluMulV(vg[k], vu[k]);
                        }
                        tk += n;
                    },
                    else => unreachable,
                }
            }
        }
    }
}

fn ffnWorker(job: *FfnJob, _: usize, _: usize) void {
    const ffn_rows = job.gate.rows;
    if (job.gate.x8 and job.up.x8) {
        switch (job.gate.typ) {
            inline .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q4_k, .q5_k, .q6_k => |gt| {
                if (job.up.typ == gt) {
                    ffnWorkerFusedX8(gt, job);
                    return;
                }
            },
            else => {},
        }
    }
    var g: [max_rows_chunk * tok_tile]f32 = undefined;
    var u: [max_rows_chunk * tok_tile]f32 = undefined;
    while (job.chunks.grab()) |range| {
        const r0 = range[0] * job.rows_chunk;
        const r1 = @min(range[1] * job.rows_chunk, ffn_rows);
        const nr = r1 - r0;
        var ts: usize = 0;
        while (ts < job.ntok) : (ts += tok_tile) {
            const te = @min(ts + tok_tile, job.ntok);
            rowsDispatch(false, job.gate, r0, r1, job.act, job.tok_offset + ts, job.tok_offset + te, &g, nr);
            rowsDispatch(false, job.up, r0, r1, job.act, job.tok_offset + ts, job.tok_offset + te, &u, nr);
            for (ts..te) |tk| {
                const gi = g[(tk - ts) * nr ..][0..nr];
                const ui = u[(tk - ts) * nr ..][0..nr];
                const o = job.out[tk * ffn_rows + r0 ..][0..nr];
                if (job.gelu) gelu_mul(o, gi, ui) else silu_mul(o, gi, ui);
            }
        }
    }
}

const CoarseJob = struct {
    coarse: [*]const u8,
    group_bytes: usize,
    act: [*]const quant.Q8B,
    nb: usize,
    sum_s: f32,
    scores: []f32,
    chunks: Chunks,
};

/// Score every 8-row group by the best of its rows' 2-bit coarse logits.
fn coarseWorker(job: *CoarseJob, _: usize, _: usize) void {
    while (job.chunks.grab()) |range| for (range[0]..range[1]) |g| {
        job.scores[g] = @reduce(.Max, repack.coarseScores(job.coarse + g * job.group_bytes, job.act, job.nb, job.sum_s));
    };
}

const PrunedFineJob = struct {
    w: *const Tensor,
    act_ptr: [*]const u8,
    cands: []const u32,
    logits: [*]f32,
    chunks: Chunks,
};

fn prunedFineWorker(job: *PrunedFineJob, _: usize, _: usize) void {
    const w = job.w;
    const typ = w.typ;
    const acts = [_][*]const u8{job.act_ptr};
    while (job.chunks.grab()) |range| {
        for (range[0]..range[1]) |i| {
            const g = job.cands[i];
            const group_ptr = w.data + @as(usize, g) * w.row_bytes;
            const v8 = switch (typ) {
                inline .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q4_k, .q5_k, .q6_k => |t| repack.gemm(t, 1, group_ptr, acts, w.cols)[0],
                else => unreachable,
            };
            const out: *[8]f32 = job.logits[g * 8 ..][0..8];
            out.* = v8;
        }
    }
}

const V8f = @Vector(8, f32);
const V8i32 = @Vector(8, i32);


fn rmsnorm(out: []f32, x: []const f32, w: ?[]const f32, eps: f32) void {
    var ss: V8f = @splat(0);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) {
        const v: V8f = x[i..][0..8].*;
        ss += v * v;
    }
    var s = @reduce(.Add, ss);
    while (i < x.len) : (i += 1) s += x[i] * x[i];
    const scale = 1.0 / @sqrt(s / @as(f32, @floatFromInt(x.len)) + eps);
    if (w) |weight| {
        for (out, x, weight) |*o, v, g| o.* = v * scale * g;
    } else {
        for (out, x) |*o, v| o.* = v * scale;
    }
}

fn fusedRmsnormQ8B(out_f: []f32, out_q: []quant.Q8B, x: []const f32, w: []const f32, eps: f32) void {
    var ss: V8f = @splat(0);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) {
        const v: V8f = x[i..][0..8].*;
        ss += v * v;
    }
    var s = @reduce(.Add, ss);
    while (i < x.len) : (i += 1) s += x[i] * x[i];
    const scale = 1.0 / @sqrt(s / @as(f32, @floatFromInt(x.len)) + eps);
    const scale_v: V8f = @splat(scale);

    const nb = x.len / 32;
    std.debug.assert(out_q.len >= nb);
    for (0..nb) |bi| {
        const off = bi * 32;
        const b = &out_q[bi];
        var v: [4]V8f = undefined;
        var amax: V8f = @splat(0);
        inline for (0..4) |j| {
            const xv: V8f = x[off + j * 8 ..][0..8].*;
            const wv: V8f = w[off + j * 8 ..][0..8].*;
            const yv = xv * scale_v * wv;
            v[j] = yv;
            out_f[off + j * 8 ..][0..8].* = yv;
            amax = @max(amax, @abs(yv));
        }
        const m = @reduce(.Max, amax);
        if (m == 0) {
            b.* = std.mem.zeroes(quant.Q8B);
            continue;
        }
        const id: V8f = @splat(127.0 / m);
        b.d = m / 127.0;
        var sum: V8i32 = @splat(0);
        inline for (0..4) |j| {
            const q: V8i32 = @intFromFloat(@round(v[j] * id));
            sum += q;
            b.qs[j * 8 ..][0..8].* = @as(@Vector(8, i8), @intCast(q));
        }
        b.s = b.d * @as(f32, @floatFromInt(@reduce(.Add, sum)));
    }
}

fn fusedRmsnormQ8K(out_f: []f32, out_q: []quant.Q8K, x: []const f32, w: []const f32, eps: f32) void {
    var ss: V8f = @splat(0);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) {
        const v: V8f = x[i..][0..8].*;
        ss += v * v;
    }
    var s = @reduce(.Add, ss);
    while (i < x.len) : (i += 1) s += x[i] * x[i];
    const scale = 1.0 / @sqrt(s / @as(f32, @floatFromInt(x.len)) + eps);
    const scale_v: V8f = @splat(scale);

    const nb = x.len / 256;
    std.debug.assert(out_q.len >= nb);
    for (0..nb) |bi| {
        const off = bi * 256;
        const b = &out_q[bi];
        var amax0: V8f = @splat(0);
        var amax1: V8f = @splat(0);
        var amax2: V8f = @splat(0);
        var amax3: V8f = @splat(0);
        inline for (0..8) |j| {
            inline for (0..4) |k| {
                const idx = j * 4 + k;
                const xv: V8f = x[off + idx * 8 ..][0..8].*;
                const wv: V8f = w[off + idx * 8 ..][0..8].*;
                const yv = xv * scale_v * wv;
                out_f[off + idx * 8 ..][0..8].* = yv;
                switch (k) {
                    0 => amax0 = @max(amax0, @abs(yv)),
                    1 => amax1 = @max(amax1, @abs(yv)),
                    2 => amax2 = @max(amax2, @abs(yv)),
                    3 => amax3 = @max(amax3, @abs(yv)),
                    else => unreachable,
                }
            }
        }
        const amax = @max(@max(amax0, amax1), @max(amax2, amax3));
        const m = @reduce(.Max, amax);
        if (m == 0) {
            b.* = std.mem.zeroes(quant.Q8K);
            continue;
        }
        const id: V8f = @splat(127.0 / m);
        b.d = m / 127.0;
        inline for (0..16) |j| {
            const v0: V8f = out_f[off + j * 16 ..][0..8].*;
            const v1: V8f = out_f[off + j * 16 + 8 ..][0..8].*;
            const q0: V8i32 = @intFromFloat(@round(v0 * id));
            const q1: V8i32 = @intFromFloat(@round(v1 * id));
            b.qs[j * 16 ..][0..8].* = @as(@Vector(8, i8), @intCast(q0));
            b.qs[j * 16 + 8 ..][0..8].* = @as(@Vector(8, i8), @intCast(q1));
            b.bsums[j] = @intCast(@reduce(.Add, q0 + q1));
        }
        inline for (0..8) |j| b.bsums32[j] = b.bsums[2 * j] + b.bsums[2 * j + 1];
    }
}

// ---------------------------------------------------------------------------
// Attention
//
// K is cached transposed in tiles of 16 positions ([head_dim][16] f16 per
// tile) so a score block is a tiny GEMM: each K tile is converted to f32 once
// and reused by up to 4 query vectors (GQA heads and/or prompt tokens) that
// share the KV head. V stays row-major. Softmax is computed online per tile
// (flash-attention style), so no score buffers scale with context length.
//
// With `--cache q8_0` both caches shrink to q8_0: per 32 head dims, one
// max-abs scale per position (f16). K tiles become [head_dim/32][32 i8][16]
// with the 16 scales after each block (hd*17 bytes per tile); V keeps
// row-major values plus tile-grouped scales after the 16 rows (hd*17 per
// tile). Attention dequantizes tiles to f32 on the fly; V scales are folded
// into the softmax weights once per block.

// KV cache packing, FWHT, PMM scales, and quantization are implemented in attention.zig


const AttnJob = struct {
    st: *State,
    slot: usize,
    n_kv: usize,
    /// Sliding window length, 0 = full causal attention.
    window: usize,
    w_tiles: usize,
    pos0: usize,
    ntok: usize,
    block_q: usize,
    /// Query blocks per KV head.
    qblocks: usize,
    flash_chunks: usize = 1,
    chunk_tiles: usize = 0,
    tile_start: usize = 0,
    head_dim: usize,
    q_dim: usize,
    attn_scale: f32,
    chunks: Chunks,
};

fn attnWorker(job: *AttnJob, _: usize, _: usize) void {
    const st = job.st;
    const c = &st.model.cfg;
    const hd = job.head_dim;
    const q_dim = job.q_dim;
    const g = c.n_heads / job.n_kv;
    const scale = job.attn_scale;
    const nq_total = job.ntok * g;
    const bq = job.block_q;
    const is_flash = job.flash_chunks > 1;
    const qb_total = job.qblocks * job.flash_chunks;

    while (job.chunks.grab()) |range| for (range[0]..range[1]) |item| {
        const kh: usize = if (is_flash) item / qb_total else item / job.qblocks;
        const rem: usize = if (is_flash) item % qb_total else item % job.qblocks;
        const qb: usize = if (is_flash) rem / job.flash_chunks else rem;
        const c_idx: usize = if (is_flash) rem % job.flash_chunks else 0;

        const first = qb * bq;
        const nq = @min(bq, nq_total - first);
        const kt = st.k_cache.ptr + st.cacheIndex(job.slot, kh);
        const vr = st.v_cache.ptr + st.vCacheIndex(job.slot, kh);

        const t_start = if (is_flash) job.tile_start + c_idx * job.chunk_tiles else 0;
        const t_end = if (is_flash) t_start + job.chunk_tiles else std.math.maxInt(usize);

        switch (nq) {
            inline 1...max_q => |n| {
                var qp: [n][*]const f32 = undefined;
                var lo: [n]usize = undefined;
                var np: [n]usize = undefined;
                var op: [n][*]f32 = undefined;
                inline for (0..n) |i| {
                    // Queries of one KV head ordered (token, head in group).
                    const t = (first + i) / g;
                    const h = kh * g + (first + i) % g;
                    qp[i] = st.q.ptr + t * q_dim + h * hd;
                    np[i] = job.pos0 + t + 1;
                    lo[i] = if (job.window > 0 and np[i] > job.window) np[i] - job.window else 0;
                    op[i] = if (is_flash)
                        st.flash_o.ptr + c_idx * q_dim + h * hd
                    else
                        st.att.f.ptr + t * q_dim + h * hd;
                }
                if (is_flash) {
                    var ps: PartialStats = undefined;
                    switch (st.cache_type) {
                        .f16 => attendBlockChunk(n, .f16, hd, kt, vr, qp, lo, np, op, scale, job.window, job.w_tiles, t_start, t_end, &ps),
                        .q8_0 => attendBlockChunk(n, .q8_0, hd, kt, vr, qp, lo, np, op, scale, job.window, job.w_tiles, t_start, t_end, &ps),
                        .q4_0 => attendBlockChunk(n, .q4_0, hd, kt, vr, qp, lo, np, op, scale, job.window, job.w_tiles, t_start, t_end, &ps),
                        .q2_0 => attendBlockChunk(n, .q2_0, hd, kt, vr, qp, lo, np, op, scale, job.window, job.w_tiles, t_start, t_end, &ps),
                        .q2_k => attendBlockChunk(n, .q2_k, hd, kt, vr, qp, lo, np, op, scale, job.window, job.w_tiles, t_start, t_end, &ps),
                    }
                    inline for (0..n) |i| {
                        const h = kh * g + (first + i) % g;
                        st.flash_m[c_idx * c.n_heads + h] = ps.m[i];
                        st.flash_l[c_idx * c.n_heads + h] = ps.l[i];
                    }
                } else {
                    switch (st.cache_type) {
                        .f16 => attendBlockChunk(n, .f16, hd, kt, vr, qp, lo, np, op, scale, job.window, job.w_tiles, 0, std.math.maxInt(usize), null),
                        .q8_0 => attendBlockChunk(n, .q8_0, hd, kt, vr, qp, lo, np, op, scale, job.window, job.w_tiles, 0, std.math.maxInt(usize), null),
                        .q4_0 => attendBlockChunk(n, .q4_0, hd, kt, vr, qp, lo, np, op, scale, job.window, job.w_tiles, 0, std.math.maxInt(usize), null),
                        .q2_0 => attendBlockChunk(n, .q2_0, hd, kt, vr, qp, lo, np, op, scale, job.window, job.w_tiles, 0, std.math.maxInt(usize), null),
                        .q2_k => attendBlockChunk(n, .q2_k, hd, kt, vr, qp, lo, np, op, scale, job.window, job.w_tiles, 0, std.math.maxInt(usize), null),
                    }
                }
            },
            else => unreachable,
        }
    };
}

fn reduceFlashDecoding(s: *State, n_chunks: usize) void {
    const c = &s.model.cfg;
    attention.reduceFlashDecoding(s.att.f.ptr, s.flash_o.ptr, s.flash_m, s.flash_l, c.n_heads, c.head_dim, c.q_dim, n_chunks);
}

pub const State = struct {
    gpa: Allocator,
    model: *const Model,
    pool: *Pool,
    ctx: usize,
    bmax: usize,
    /// Tokens currently held in the KV cache.
    pos: usize = 0,

    x: []f32,
    xn: Act,
    att: Act,
    hb: Act,
    q: []f32,
    k: []f32,
    v: []f32,
    logits: []f32,
    rope_cs: []f32,
    rope_cs_swa: []f32,
    /// NTK-aware scaled inverse frequencies when context extends beyond training window
    inv_freq: []f32 = &.{},
    /// Per (attention layer, kv head) byte region: K and V in tiles of 16
    /// positions. f16: hd*16 f16 values per tile (V row-major). q8_0: hd*17
    /// bytes per tile (32 i8 values + 16 f16 scales per 32 head dims).
    /// q4_0: hd*9 bytes per tile (32 4-bit values + 16 f16 scales per 32 head dims).
    k_cache: []align(32) u8,
    v_cache: []align(32) u8,
    /// Byte offsets into k_cache / v_cache for each attention layer slot.
    slot_offsets: []usize,
    /// Bytes of k_cache / v_cache per KV head for each attention layer slot.
    slot_strides: []usize,
    v_slot_offsets: []usize,
    v_slot_strides: []usize,
    /// Number of 16-token tiles allocated per KV head for each attention layer slot.
    slot_tiles: []usize,
    /// K/V cache format: f16, q8_0, or q4_0.
    cache_type: CacheType,
    /// ctx rounded up to whole K tiles.
    ctx_pad: usize,
    /// Mixer / FFN output awaiting its post-norm (sandwich-norm models).
    tmp: []f32,
    /// Per-head attention gate logits [bmax][n_heads].
    gate: []f32,
    /// FlashDecoding partial outputs and accumulators [max_flash_chunks][q_dim]
    flash_o: []f32,
    flash_m: []f32,
    flash_l: []f32,
    /// MoE router logits and combined expert output
    router_logits: []f32,
    moe_out: []f32,
    /// Gemma 4 Per-Layer Embeddings (PLE)
    inp_per_layer: []f32 = &.{},
    ple_proj: []f32 = &.{},
    ple_buf: []f32 = &.{},
    ple_act: Act = undefined,
    /// Short conv: in_proj output [bmax][3*dim], input history
    /// [(kernel-1) + bmax][dim], and the last kernel-1 inputs of every conv layer.
    conv_in: []f32,
    conv_hist: []f32,
    conv_state: []f32,
    conv_checkpoint: []f32,
    coarse_scores: []f32 = &.{},
    cands: []u32 = &.{},

    pub fn init(gpa: Allocator, model: *const Model, pool: *Pool, ctx: usize, bmax: usize, cache_type: CacheType) !State {
        const c = &model.cfg;
        if (c.head_dim % 8 != 0 or c.head_dim > max_head_dim) return error.UnsupportedHeadDim;
        if (cache_type != .f16 and c.head_dim % 32 != 0) return error.UnsupportedHeadDim;
        var s: State = undefined;
        s.gpa = gpa;
        s.model = model;
        s.pool = pool;
        s.ctx = ctx;
        const batch_cap = @max(bmax, max_spec + 1);
        s.bmax = batch_cap;
        s.pos = 0;
        s.cache_type = cache_type;
        s.ctx_pad = std.mem.alignForward(usize, ctx, kv_tile);
        const hist = if (c.conv_kernel > 0) c.conv_kernel - 1 else 0;
        const has_conv = model.n_conv_layers > 0;
        s.xn = try Act.init(gpa, c.dim, batch_cap);
        errdefer s.xn.deinit(gpa);
        s.att = try Act.init(gpa, c.q_dim, batch_cap);
        errdefer s.att.deinit(gpa);
        s.hb = try Act.init(gpa, @max(c.ffn_dim, c.expert_feed_forward_length), batch_cap);
        errdefer s.hb.deinit(gpa);
        // Plain f32 buffers: one allocation each, freed together on failure.
        const bufs = .{
            .{ "x", batch_cap * c.dim },
            .{ "q", batch_cap * c.q_dim },
            .{ "k", batch_cap * c.kv_dim },
            .{ "v", batch_cap * c.kv_dim },
            .{ "logits", (max_spec + 1) * c.vocab },
            .{ "rope_cs", batch_cap * c.n_rot },
            .{ "rope_cs_swa", if (model.inv_freq_swa.len > 0) batch_cap * (if (c.arch == .gemma4) c.n_rot_swa else c.n_rot) else 0 },
            .{ "tmp", batch_cap * c.dim },
            .{ "gate", batch_cap * c.n_heads },
            .{ "conv_in", if (has_conv) batch_cap * 3 * c.dim else 0 },
            .{ "conv_hist", if (has_conv) (hist + batch_cap) * c.dim else 0 },
            .{ "conv_state", model.n_conv_layers * hist * c.dim },
            .{ "conv_checkpoint", if (has_conv) (max_spec + 1) * model.n_conv_layers * hist * c.dim else 0 },
            .{ "flash_o", max_flash_chunks * c.q_dim },
            .{ "flash_m", max_flash_chunks * c.n_heads },
            .{ "flash_l", max_flash_chunks * c.n_heads },
            .{ "router_logits", if (c.n_experts > 0) batch_cap * c.n_experts else 0 },
            .{ "moe_out", if (c.n_experts > 0) batch_cap * c.dim else 0 },
            .{ "inp_per_layer", if (c.n_embd_per_layer > 0) batch_cap * c.n_layers * c.n_embd_per_layer else 0 },
            .{ "ple_proj", if (c.n_embd_per_layer > 0) batch_cap * c.n_layers * c.n_embd_per_layer else 0 },
            .{ "ple_buf", if (c.n_embd_per_layer > 0) batch_cap * c.n_embd_per_layer else 0 },
        };
        var done: usize = 0;
        errdefer inline for (bufs, 0..) |b, i| {
            if (i < done) gpa.free(@field(s, b[0]));
        };
        inline for (bufs) |b| {
            @field(s, b[0]) = try gpa.alloc(f32, b[1]);
            done += 1;
        }
        if (c.n_embd_per_layer > 0) {
            s.ple_act = try Act.init(gpa, c.n_embd_per_layer, batch_cap);
        }
        @memset(s.conv_state, 0);

        s.slot_offsets = try gpa.alloc(usize, model.n_attn_layers);
        errdefer gpa.free(s.slot_offsets);
        s.slot_strides = try gpa.alloc(usize, model.n_attn_layers);
        errdefer gpa.free(s.slot_strides);
        s.v_slot_offsets = try gpa.alloc(usize, model.n_attn_layers);
        errdefer gpa.free(s.v_slot_offsets);
        s.v_slot_strides = try gpa.alloc(usize, model.n_attn_layers);
        errdefer gpa.free(s.v_slot_strides);
        s.slot_tiles = try gpa.alloc(usize, model.n_attn_layers);
        errdefer gpa.free(s.slot_tiles);

        var total_k_bytes: usize = 0;
        var total_v_bytes: usize = 0;
        for (model.layers) |*L| {
            switch (L.mixer) {
                .attention => |*A| {
                    if (A.has_kv) {
                        const is_swa = A.swa and c.swa_window > 0;
                        const tiles = if (is_swa) (c.swa_window + kv_tile - 1) / kv_tile + 1 else s.ctx_pad / kv_tile;
                        const k_tile_bytes: usize = switch (cache_type) {
                            .f16 => A.head_dim * 32,
                            .q8_0 => A.head_dim * 17,
                            .q4_0 => A.head_dim * 9,
                            .q2_0, .q2_k => A.head_dim * 5,
                        };
                        const v_tile_bytes: usize = switch (cache_type) {
                            .f16 => A.head_dim * 32,
                            .q8_0 => A.head_dim * 17,
                            .q4_0, .q2_k => A.head_dim * 9,
                            .q2_0 => A.head_dim * 5,
                        };
                        const k_head_bytes = tiles * k_tile_bytes;
                        const v_head_bytes = tiles * v_tile_bytes;
                        s.slot_tiles[A.kv_slot] = tiles;
                        s.slot_strides[A.kv_slot] = k_head_bytes;
                        s.slot_offsets[A.kv_slot] = total_k_bytes;
                        s.v_slot_strides[A.kv_slot] = v_head_bytes;
                        s.v_slot_offsets[A.kv_slot] = total_v_bytes;
                        total_k_bytes += c.n_kv_heads * k_head_bytes;
                        total_v_bytes += c.n_kv_heads * v_head_bytes;
                    }
                },
                .shortconv => {},
            }
        }

        s.k_cache = try gpa.alignedAlloc(u8, .@"32", total_k_bytes);
        errdefer gpa.free(s.k_cache);
        s.v_cache = try gpa.alignedAlloc(u8, .@"32", total_v_bytes);
        if (model.output_coarse != null) {
            const n_groups = c.vocab / 8;
            s.coarse_scores = try gpa.alloc(f32, n_groups);
            errdefer gpa.free(s.coarse_scores);
            s.cands = try gpa.alloc(u32, 512 + 64);
            errdefer gpa.free(s.cands);
        } else {
            s.coarse_scores = &.{};
            s.cands = &.{};
        }

        // NTK-aware scaled RoPE for extended context beyond training length
        if (s.ctx > c.ctx_train and c.ctx_train > 0 and c.n_rot > 2 and model.inv_freq.len > 0) {
            const base_ntk = rope_mod.ntkAwareBase(c.rope_base, s.ctx, c.ctx_train, c.n_rot);
            s.inv_freq = try gpa.alloc(f32, model.inv_freq.len);
            rope_mod.ropeFreqs(base_ntk, c.rope_scale, null, s.inv_freq);
        } else {
            s.inv_freq = &.{};
        }

        return s;
    }

    pub fn deinit(s: *State) void {
        const gpa = s.gpa;
        s.xn.deinit(gpa);
        s.att.deinit(gpa);
        s.hb.deinit(gpa);
        if (s.model.cfg.n_embd_per_layer > 0) s.ple_act.deinit(gpa);
        inline for (.{ "x", "q", "k", "v", "logits", "rope_cs", "rope_cs_swa", "tmp", "gate", "conv_in", "conv_hist", "conv_state", "conv_checkpoint", "flash_o", "flash_m", "flash_l", "router_logits", "moe_out", "inp_per_layer", "ple_proj", "ple_buf" }) |name| gpa.free(@field(s, name));
        gpa.free(s.slot_offsets);
        gpa.free(s.slot_strides);
        gpa.free(s.v_slot_offsets);
        gpa.free(s.v_slot_strides);
        gpa.free(s.slot_tiles);
        gpa.free(s.k_cache);
        gpa.free(s.v_cache);
        if (s.coarse_scores.len > 0) gpa.free(s.coarse_scores);
        if (s.cands.len > 0) gpa.free(s.cands);
        if (s.inv_freq.len > 0) gpa.free(s.inv_freq);
    }
    /// Forget the context (KV cache positions and convolution state).
    pub fn reset(s: *State) void {
        s.pos = 0;
        @memset(s.conv_state, 0);
        @memset(s.conv_checkpoint, 0);
    }

    inline fn convCheckpoint(s: *State, t: usize, slot: usize) []f32 {
        const hist = s.model.cfg.conv_kernel - 1;
        const d = s.model.cfg.dim;
        const layer_sz = hist * d;
        const step_sz = s.model.n_conv_layers * layer_sz;
        return s.conv_checkpoint[t * step_sz + slot * layer_sz ..][0..layer_sz];
    }

    /// Rewind KV cache position and convolution state after speculative rejection.
    pub fn rewind(s: *State, kept: usize, batch_len: usize) void {
        std.debug.assert(kept <= batch_len);
        s.pos -= (batch_len - kept);
        if (s.model.n_conv_layers > 0 and s.conv_checkpoint.len > 0 and batch_len <= max_spec) {
            const hist = s.model.cfg.conv_kernel - 1;
            const d = s.model.cfg.dim;
            const step_sz = s.model.n_conv_layers * hist * d;
            const src = s.conv_checkpoint[kept * step_sz ..][0..s.conv_state.len];
            @memcpy(s.conv_state, src);
        }
    }

    /// Short-conv state after the first `t` tokens of the last forward batch;
    /// checkpointed only for batches of at most `max_spec` tokens. Empty for
    /// models without short-conv layers.
    pub fn convCheckpointState(s: *const State, t: usize) []const f32 {
        return s.conv_checkpoint[t * s.conv_state.len ..][0..s.conv_state.len];
    }

    /// Keep only the first `pos` cached tokens. `conv` is the short-conv state
    /// at exactly that position (empty for models without short-conv layers);
    /// attention entries past `pos` are overwritten before they are read.
    pub fn truncate(s: *State, pos: usize, conv: []const f32) void {
        std.debug.assert(pos <= s.pos and conv.len == s.conv_state.len);
        s.pos = pos;
        @memcpy(s.conv_state, conv);
    }

    /// Start of one (attention layer, kv head) region in k_cache.
    pub inline fn cacheIndex(s: *const State, slot: usize, kh: usize) usize {
        return s.slot_offsets[slot] + kh * s.slot_strides[slot];
    }

    /// Start of one (attention layer, kv head) region in v_cache.
    pub inline fn vCacheIndex(s: *const State, slot: usize, kh: usize) usize {
        return s.v_slot_offsets[slot] + kh * s.v_slot_strides[slot];
    }

    fn matmulOffset(s: *State, outs: []const MatOut, act: *Act, tok_offset: usize, ntok: usize) void {
        std.debug.assert(outs.len <= max_outs);
        var rows: usize = 0;
        for (outs) |o| {
            act.prepare(quant.actKind(o.w.typ), s.pool);
            rows += o.w.rows;
        }
        const rc = rowsChunk(rows, s.pool.nth);
        var job: MatJob = .{ .outs = outs, .act = act, .ntok = ntok, .tok_offset = tok_offset, .rows_chunk = rc, .first = undefined, .chunks = undefined };
        job.first[0] = 0;
        for (outs, 0..) |o, i| job.first[i + 1] = job.first[i] + (o.w.rows + rc - 1) / rc;
        job.chunks = .init(job.first[outs.len], 1);
        s.pool.run(&job, matWorker);
    }

    fn matmul(s: *State, outs: []const MatOut, act: *Act) void {
        s.matmulOffset(outs, act, 0, act.ntok);
    }

    /// Rows per work item: a multiple of 8 (x8 groups), ~8 items per thread.
    fn rowsChunk(rows: usize, nth: usize) usize {
        return std.math.clamp(rows / (nth * 8) / 8 * 8, 8, max_rows_chunk);
    }

    fn ropeTables(s: *State, pos0: usize, ntok: usize) void {
        const m = s.model;
        const n_rot = m.cfg.n_rot;
        const freqs = if (s.inv_freq.len > 0) s.inv_freq else m.inv_freq;
        if (freqs.len > 0) {
            rope_mod.precomputeRopeTable(s.rope_cs, freqs, pos0, ntok, n_rot);
        }
        if (m.inv_freq_swa.len > 0) {
            const n_rot_swa = if (m.cfg.arch == .gemma4) m.cfg.n_rot_swa else n_rot;
            rope_mod.precomputeRopeTable(s.rope_cs_swa, m.inv_freq_swa, pos0, ntok, n_rot_swa);
        }
    }

    fn rope(s: *const State, v: []f32, t: usize, swa: bool, hd: usize, n_rot: usize) void {
        const c = &s.model.cfg;
        const cs = (if (swa) s.rope_cs_swa[t * (if (c.arch == .gemma4) c.n_rot_swa else c.n_rot) ..] else s.rope_cs[t * c.n_rot ..]);
        rope_mod.applyRope(v, cs, hd, n_rot, c.rope_neox);
    }

    /// Attention mixer for `nt` tokens of normalized input in `xn`; the
    /// projected output is written to (or accumulated into) `dst`.
    fn attention(s: *State, A: *const Attention, nt: usize, pos0: usize, dst: [*]f32, accumulate: bool) void {
        const c = &s.model.cfg;
        const hd = A.head_dim;
        const q_dim = A.q_dim;
        const kv_dim = A.kv_dim;
        var outs: [max_outs]MatOut = undefined;
        var n_outs: usize = 0;
        outs[n_outs] = .{ .w = &A.wq, .out = s.q.ptr, .stride = q_dim };
        n_outs += 1;
        if (A.has_kv) {
            outs[n_outs] = .{ .w = &A.wk, .out = s.k.ptr, .stride = kv_dim };
            n_outs += 1;
            outs[n_outs] = .{ .w = &A.wv, .out = s.v.ptr, .stride = kv_dim };
            n_outs += 1;
        }
        if (A.gate) |*g| {
            outs[n_outs] = .{ .w = g, .out = s.gate.ptr, .stride = c.n_heads };
            n_outs += 1;
        }
        s.matmul(outs[0..n_outs], &s.xn);

        for (0..nt) |t| {
            const q = s.q[t * q_dim ..][0..q_dim];
            if (A.bq) |b| for (q, b) |*a, bb| {
                a.* += bb;
            };
            if (A.q_norm) |w| {
                var h: usize = 0;
                while (h < q.len) : (h += hd) rmsnorm(q[h..][0..hd], q[h..][0..hd], w, c.eps);
            }
            s.rope(q, t, A.swa, hd, A.n_rot);
            if ((s.cache_type == .q4_0 or s.cache_type == .q2_0 or s.cache_type == .q2_k) and (hd & (hd - 1)) == 0) {
                var h: usize = 0;
                while (h < q.len) : (h += hd) fwhtSimd(q[h..][0..hd]);
            }

            if (A.has_kv) {
                const k = s.k[t * kv_dim ..][0..kv_dim];
                const v = s.v[t * kv_dim ..][0..kv_dim];
                if (A.bk) |b| for (k, b) |*a, bb| {
                    a.* += bb;
                };
                if (A.bv) |b| for (v, b) |*a, bb| {
                    a.* += bb;
                };
                if (A.k_norm) |w| {
                    var h: usize = 0;
                    while (h < k.len) : (h += hd) rmsnorm(k[h..][0..hd], k[h..][0..hd], w, c.eps);
                }
                s.rope(k, t, A.swa, hd, A.n_rot);
                if ((s.cache_type == .q4_0 or s.cache_type == .q2_0 or s.cache_type == .q2_k) and (hd & (hd - 1)) == 0) {
                    var kh: usize = 0;
                    while (kh < k.len) : (kh += hd) fwhtSimd(k[kh..][0..hd]);
                }
                const is_swa = A.swa and c.swa_window > 0;
                const w_tiles = s.slot_tiles[A.kv_slot];
                const p = pos0 + t;
                const tile = if (is_swa) (p / kv_tile) % w_tiles else p / kv_tile;
                const pl = p % kv_tile;
                const tb_k: usize = switch (s.cache_type) {
                    .f16 => hd * 32,
                    .q8_0 => hd * 17,
                    .q4_0 => hd * 9,
                    .q2_0, .q2_k => hd * 5,
                };
                const tb_v: usize = switch (s.cache_type) {
                    .f16 => hd * 32,
                    .q8_0 => hd * 17,
                    .q4_0, .q2_k => hd * 9,
                    .q2_0 => hd * 5,
                };
                for (0..A.n_kv_heads) |kh| {
                    const base_k = s.cacheIndex(A.kv_slot, kh);
                    const base_v = s.vCacheIndex(A.kv_slot, kh);
                    switch (s.cache_type) {
                        .q2_0 => {
                            const kt = s.k_cache.ptr + base_k + tile * tb_k;
                            const vt = s.v_cache.ptr + base_v + tile * tb_v;
                            const kq = k[kh * hd ..][0..hd];
                            const vq = v[kh * hd ..][0..hd];
                            const byte_idx = pl % 4;
                            const shift: u3 = @intCast((pl / 4) * 2);
                            const mask = ~(@as(u8, 0x03) << shift);

                            var b: usize = 0;
                            while (b < hd / 32) : (b += 1) {
                                var q2_k: [32]u8 = undefined;
                                var sc_k: f16 = undefined;
                                quantBlock32_q2(kq[b * 32 ..][0..32], &q2_k, &sc_k);

                                const k_bp = kt + b * 160;
                                if (pl == 0) @memset(k_bp[0..128], 0);
                                @as(*align(1) f16, @ptrCast(k_bp + 128 + pl * 2)).* = sc_k;
                                inline for (0..32) |dp| {
                                    k_bp[dp * 4 + byte_idx] = (k_bp[dp * 4 + byte_idx] & mask) | (q2_k[dp] << shift);
                                }

                                var q2_v: [32]u8 = undefined;
                                var sc_v: f16 = undefined;
                                quantBlock32_q2(vq[b * 32 ..][0..32], &q2_v, &sc_v);

                                @as(*align(1) f16, @ptrCast(vt + 4 * hd + b * 32 + pl * 2)).* = sc_v;
                                const v_dst = vt + pl * (hd / 4) + b * 8;
                                const d0: @Vector(8, u8) = q2_v[0..8].*;
                                const d1: @Vector(8, u8) = q2_v[8..16].*;
                                const d2: @Vector(8, u8) = q2_v[16..24].*;
                                const d3: @Vector(8, u8) = q2_v[24..32].*;
                                const v_packed: [8]u8 = d0 | (d1 << @as(@Vector(8, u8), @splat(2))) | (d2 << @as(@Vector(8, u8), @splat(4))) | (d3 << @as(@Vector(8, u8), @splat(6)));
                                @memcpy(v_dst[0..8], &v_packed);
                            }
                        },
                        .q2_k => {
                            const kt = s.k_cache.ptr + base_k + tile * tb_k;
                            const vt = s.v_cache.ptr + base_v + tile * tb_v;
                            const kq = k[kh * hd ..][0..hd];
                            const vq = v[kh * hd ..][0..hd];
                            const byte_idx = pl % 4;
                            const shift: u3 = @intCast((pl / 4) * 2);
                            const mask = ~(@as(u8, 0x03) << shift);

                            var b: usize = 0;
                            while (b < hd / 32) : (b += 1) {
                                var q2_k: [32]u8 = undefined;
                                var sc_k: f16 = undefined;
                                quantBlock32_q2(kq[b * 32 ..][0..32], &q2_k, &sc_k);

                                const k_bp = kt + b * 160;
                                if (pl == 0) @memset(k_bp[0..128], 0);
                                @as(*align(1) f16, @ptrCast(k_bp + 128 + pl * 2)).* = sc_k;
                                inline for (0..32) |dp| {
                                    k_bp[dp * 4 + byte_idx] = (k_bp[dp * 4 + byte_idx] & mask) | (q2_k[dp] << shift);
                                }

                                var q4_v: [32]u8 = undefined;
                                var sc_v: f16 = undefined;
                                quantBlock32_q4(vq[b * 32 ..][0..32], &q4_v, &sc_v);

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
                        },
                        .q4_0 => {
                            const kt = s.k_cache.ptr + base_k + tile * tb_k;
                            const vt = s.v_cache.ptr + base_v + tile * tb_v;
                            const kq = k[kh * hd ..][0..hd];
                            const vq = v[kh * hd ..][0..hd];
                            var b: usize = 0;
                            while (b < hd / 32) : (b += 1) {
                                var q4_k: [32]u8 = undefined;
                                var sc_k: f16 = undefined;
                                quantBlock32_q4(kq[b * 32 ..][0..32], &q4_k, &sc_k);

                                const k_bp = kt + b * 288;
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
                                quantBlock32_q4(vq[b * 32 ..][0..32], &q4_v, &sc_v);

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
                        },
                        .q8_0 => {
                            const kt = s.k_cache.ptr + base_k + tile * tb_k;
                            const vt = s.v_cache.ptr + base_v + tile * tb_v;
                            const kq = k[kh * hd ..][0..hd];
                            const vq = v[kh * hd ..][0..hd];
                            var b: usize = 0;
                            while (b < hd / 32) : (b += 1) {
                                quantBlock32(kq[b * 32 ..][0..32], kt + b * 544 + pl, kv_tile, @ptrCast(kt + b * 544 + 512 + pl * 2));
                                quantBlock32(vq[b * 32 ..][0..32], vt + pl * hd + b * 32, 1, @ptrCast(vt + 16 * hd + b * 32 + pl * 2));
                            }
                        },
                        .f16 => {
                            const kt: [*]f16 = @ptrCast(@alignCast(s.k_cache.ptr + base_k + tile * tb_k + pl * 2));
                            const vr: [*]f16 = @ptrCast(@alignCast(s.v_cache.ptr + base_v + tile * tb_v + pl * hd * 2));
                            for (0..hd) |d| {
                                kt[d * kv_tile] = @floatCast(k[kh * hd + d]);
                                vr[d] = @floatCast(v[kh * hd + d]);
                            }
                        },
                    }
                }
            }
        }

        const is_swa = A.swa and c.swa_window > 0;
        const w_tiles = s.slot_tiles[A.kv_slot];
        const g = c.n_heads / A.n_kv_heads;
        var bq: usize = max_q;
        while (bq > 1 and A.n_kv_heads * ((nt * g + bq - 1) / bq) < s.pool.nth) {
            bq /= 2;
        }
        const qblocks = (nt * g + bq - 1) / bq;
        var total_items = A.n_kv_heads * qblocks;
        var flash_chunks: usize = 1;
        var chunk_tiles: usize = 0;
        var tile_start: usize = 0;

        if (nt == 1 and s.pool.nth > 1) {
            const np = pos0 + 1;
            const lo = if (is_swa and np > c.swa_window) np - c.swa_window else 0;
            tile_start = lo / kv_tile;
            const tile_end = (np + kv_tile - 1) / kv_tile;
            const total_tiles = if (tile_end > tile_start) tile_end - tile_start else 0;
            if (total_tiles >= 32) {
                const target_tiles: usize = 16;
                flash_chunks = std.math.clamp((total_tiles + target_tiles - 1) / target_tiles, 1, max_flash_chunks);
                if (flash_chunks > 1) {
                    chunk_tiles = (total_tiles + flash_chunks - 1) / flash_chunks;
                    total_items = A.n_kv_heads * qblocks * flash_chunks;
                }
            }
        }

        const chunk = if (s.pool.nth > 1) std.math.clamp(total_items / (s.pool.nth * 4), 1, 8) else 1;
        var aj: AttnJob = .{
            .st = s,
            .slot = A.kv_slot,
            .n_kv = A.n_kv_heads,
            .window = if (is_swa) c.swa_window else 0,
            .w_tiles = w_tiles,
            .pos0 = pos0,
            .ntok = nt,
            .block_q = bq,
            .qblocks = qblocks,
            .flash_chunks = flash_chunks,
            .chunk_tiles = chunk_tiles,
            .tile_start = tile_start,
            .head_dim = A.head_dim,
            .q_dim = A.q_dim,
            .attn_scale = A.attn_scale,
            .chunks = .init(total_items, chunk),
        };
        s.pool.run(&aj, attnWorker);

        if (flash_chunks > 1) {
            reduceFlashDecoding(s, flash_chunks);
        }

        if (A.gate != null) {
            for (0..nt) |t| for (0..c.n_heads) |h| {
                const gv = 1.0 / (1.0 + @exp(-s.gate[t * c.n_heads + h]));
                for (s.att.f[t * q_dim + h * hd ..][0..hd]) |*a| a.* *= gv;
            };
        }
        s.att.reset(nt);
        s.att.n = A.q_dim;
        s.matmul(&.{.{ .w = &A.wo, .out = dst, .stride = c.dim, .accumulate = accumulate }}, &s.att);
    }

    /// LFM gated short convolution over `nt` tokens of normalized input in `xn`.
    fn shortConv(s: *State, C: *const ShortConv, nt: usize, dst: [*]f32, accumulate: bool) void {
        const c = &s.model.cfg;
        const d = c.dim;
        const taps = c.conv_kernel;
        const hist = taps - 1;
        s.matmul(&.{.{ .w = &C.in_proj, .out = s.conv_in.ptr, .stride = 3 * d }}, &s.xn);

        // History rows: the previous `hist` conv inputs, then B*x of each new token.
        const state = s.conv_state[C.slot * hist * d ..][0 .. hist * d];
        @memcpy(s.conv_hist[0 .. hist * d], state);
        for (0..nt) |t| {
            const bcx = s.conv_in[t * 3 * d ..];
            const row = s.conv_hist[(hist + t) * d ..][0..d];
            for (row, bcx[0..d], bcx[2 * d ..][0..d]) |*r, b, x| r.* = b * x;
        }
        // y = C * sum_k w[k] * hist[t + k], written straight into xn for out_proj.
        for (0..nt) |t| {
            const cg = s.conv_in[t * 3 * d + d ..][0..d];
            const y = s.xn.f[t * d ..][0..d];
            var ch: usize = 0;
            while (ch < d) : (ch += 8) {
                var acc: V8f = @splat(0);
                for (0..taps) |k| acc = @mulAdd(V8f, C.kernel[k * d + ch ..][0..8].*, s.conv_hist[(t + k) * d + ch ..][0..8].*, acc);
                y[ch..][0..8].* = acc * @as(V8f, cg[ch..][0..8].*);
            }
        }
        @memcpy(state, s.conv_hist[nt * d ..][0 .. hist * d]);
        if (s.conv_checkpoint.len > 0 and nt <= max_spec) {
            for (0..nt + 1) |t| {
                @memcpy(s.convCheckpoint(t, C.slot), s.conv_hist[t * d ..][0 .. hist * d]);
            }
        }
        s.xn.reset(nt);
        s.matmul(&.{.{ .w = &C.out_proj, .out = dst, .stride = d, .accumulate = accumulate }}, &s.xn);
    }

    /// x += rmsnorm(tmp) * w for every token (sandwich norm residual).
    fn addNormed(s: *State, w: []const f32, nt: usize) void {
        const c = &s.model.cfg;
        for (0..nt) |t| {
            const y = s.tmp[t * c.dim ..][0..c.dim];
            rmsnorm(y, y, w, c.eps);
            for (s.x[t * c.dim ..][0..c.dim], y) |*a, b| a.* += b;
        }
    }
    fn outputCoarseToFine(s: *State, t: usize) void {
        const m = s.model;
        const w = &m.output;
        const typ = w.typ;
        const n_groups = m.cfg.vocab / 8;

        const kind = quant.actKind(typ);
        s.xn.prepare(kind, s.pool);
        const act_ptr: [*]const u8 = switch (kind) {
            .q8k => s.xn.rowPtr(.q8k, t),
            .q8b => s.xn.rowPtr(.q8b, t),
            .f32 => s.xn.rowPtr(.f32, t),
        };

        // 1. Parallel coarse pass over all groups: 2-bit weights x Q8B activations.
        s.xn.prepare(.q8b, s.pool);
        const nb = w.cols / 32;
        const q8b = s.xn.q8b[t * nb ..][0..nb];
        var sum_s: f32 = 0;
        for (q8b) |b| sum_s += b.s;
        var c_job: CoarseJob = .{
            .coarse = m.output_coarse.?.ptr,
            .group_bytes = m.output_coarse_bytes,
            .act = q8b.ptr,
            .nb = nb,
            .sum_s = sum_s,
            .scores = s.coarse_scores,
            .chunks = .init(n_groups, 64),
        };
        s.pool.run(&c_job, coarseWorker);

        // 2. Select top candidate groups with min-heap
        const cand_count = 512;
        const fixed_count = 64;
        var heap_scores: [cand_count]f32 = undefined;
        var heap_indices: [cand_count]u32 = undefined;
        var heap_size: usize = 0;

        for (s.coarse_scores, 0..) |score, i| {
            if (heap_size < cand_count) {
                heap_scores[heap_size] = score;
                heap_indices[heap_size] = @intCast(i);
                heap_size += 1;
                var curr = heap_size - 1;
                while (curr > 0) {
                    const parent = (curr - 1) / 2;
                    if (heap_scores[curr] < heap_scores[parent]) {
                        const tmp_s = heap_scores[curr];
                        heap_scores[curr] = heap_scores[parent];
                        heap_scores[parent] = tmp_s;
                        const tmp_i = heap_indices[curr];
                        heap_indices[curr] = heap_indices[parent];
                        heap_indices[parent] = tmp_i;
                        curr = parent;
                    } else break;
                }
            } else if (score > heap_scores[0]) {
                heap_scores[0] = score;
                heap_indices[0] = @intCast(i);
                var curr: usize = 0;
                while (true) {
                    const left = 2 * curr + 1;
                    const right = 2 * curr + 2;
                    var smallest = curr;
                    if (left < cand_count and heap_scores[left] < heap_scores[smallest]) smallest = left;
                    if (right < cand_count and heap_scores[right] < heap_scores[smallest]) smallest = right;
                    if (smallest != curr) {
                        const tmp_s = heap_scores[curr];
                        heap_scores[curr] = heap_scores[smallest];
                        heap_scores[smallest] = tmp_s;
                        const tmp_i = heap_indices[curr];
                        heap_indices[curr] = heap_indices[smallest];
                        heap_indices[smallest] = tmp_i;
                        curr = smallest;
                    } else break;
                }
            }
        }

        for (0..fixed_count) |i| s.cands[i] = @intCast(i);
        var actual_cands: usize = fixed_count;
        for (heap_indices) |g| {
            if (g >= fixed_count) {
                s.cands[actual_cands] = g;
                actual_cands += 1;
            }
        }

        const l_out = s.logits.ptr + t * m.cfg.vocab;
        @memset(l_out[0..m.cfg.vocab], -std.math.inf(f32));

        // 3. Parallel fine pass only on candidate groups
        var f_job: PrunedFineJob = .{
            .w = w,
            .act_ptr = act_ptr,
            .cands = s.cands[0..actual_cands],
            .logits = l_out,
            .chunks = .init(actual_cands, 32),
        };
        s.pool.run(&f_job, prunedFineWorker);
    }

    const NormJob = struct {
        s: *State,
        x: [*]const f32,
        w: []const f32,
        eps: f32,
        kind: quant.ActKind,
        chunks: Chunks,
    };

    fn normWorker(j: *NormJob, _: usize, _: usize) void {
        const c = j.s.model.cfg;
        while (j.chunks.grab()) |r| {
            for (r[0]..r[1]) |t| {
                const src_x = j.x[t * c.dim ..][0..c.dim];
                const dst_f = j.s.xn.f[t * c.dim ..][0..c.dim];
                switch (j.kind) {
                    .q8k => fusedRmsnormQ8K(dst_f, j.s.xn.q8k[t * (c.dim / 256) ..][0 .. c.dim / 256], src_x, j.w, j.eps),
                    .q8b => fusedRmsnormQ8B(dst_f, j.s.xn.q8b[t * (c.dim / 32) ..][0 .. c.dim / 32], src_x, j.w, j.eps),
                    .f32 => rmsnorm(dst_f, src_x, j.w, j.eps),
                }
            }
        }
    }

    fn fusedNorm(s: *State, nt: usize, x_src: [*]const f32, w: []const f32, kind: quant.ActKind) void {
        const c = s.model.cfg;
        s.xn.reset(nt);
        switch (kind) {
            .q8k => {
                s.xn.have_q8k = true;
                if (nt * c.dim >= 32768 and s.pool.nth > 1) {
                    var job: NormJob = .{ .s = s, .x = x_src, .w = w, .eps = c.eps, .kind = .q8k, .chunks = .init(nt, 1) };
                    s.pool.run(&job, normWorker);
                } else {
                    for (0..nt) |t| {
                        fusedRmsnormQ8K(
                            s.xn.f[t * c.dim ..][0..c.dim],
                            s.xn.q8k[t * (c.dim / 256) ..][0 .. c.dim / 256],
                            x_src[t * c.dim ..][0..c.dim],
                            w,
                            c.eps,
                        );
                    }
                }
            },
            .q8b => {
                s.xn.have_q8b = true;
                if (nt * c.dim >= 32768 and s.pool.nth > 1) {
                    var job: NormJob = .{ .s = s, .x = x_src, .w = w, .eps = c.eps, .kind = .q8b, .chunks = .init(nt, 1) };
                    s.pool.run(&job, normWorker);
                } else {
                    for (0..nt) |t| {
                        fusedRmsnormQ8B(
                            s.xn.f[t * c.dim ..][0..c.dim],
                            s.xn.q8b[t * (c.dim / 32) ..][0 .. c.dim / 32],
                            x_src[t * c.dim ..][0..c.dim],
                            w,
                            c.eps,
                        );
                    }
                }
            },
            .f32 => {
                if (nt * c.dim >= 32768 and s.pool.nth > 1) {
                    var job: NormJob = .{ .s = s, .x = x_src, .w = w, .eps = c.eps, .kind = .f32, .chunks = .init(nt, 1) };
                    s.pool.run(&job, normWorker);
                } else {
                    for (0..nt) |t| {
                        rmsnorm(
                            s.xn.f[t * c.dim ..][0..c.dim],
                            x_src[t * c.dim ..][0..c.dim],
                            w,
                            c.eps,
                        );
                    }
                }
            },
        }
    }

    fn forwardMoE(s: *State, moe: *const MoEFfn, ffn_norm: []const f32, post_ffn_norm: ?[]const f32, nt: usize) void {
        const c = &s.model.cfg;
        const dim = c.dim;
        const ffn_dim = c.expert_feed_forward_length;
        const n_exp = c.n_experts;
        const k_exp = @min(c.n_experts_used, n_exp);

        // 1. Normalize input x into s.xn
        const router_kind = quant.actKind(moe.router.typ);
        s.fusedNorm(nt, s.x.ptr, ffn_norm, router_kind);

        // 2. Compute router logits: [nt][n_exp]
        s.matmul(&.{.{ .w = &moe.router, .out = s.router_logits.ptr, .stride = n_exp }}, &s.xn);

        // Clear moe_out buffer for the batch
        @memset(s.moe_out[0 .. nt * dim], 0);

        // Top-K routing buffers (supports up to max_active_experts active experts per token)
        if (k_exp > max_active_experts) return;
        var top_idx: [max_active_experts]usize = undefined;
        var top_val: [max_active_experts]f32 = undefined;

        for (0..nt) |t| {
            const logits = s.router_logits[t * n_exp ..][0..n_exp];

            // 3. Find top-K experts and softmax weights for token t
            ffn.routeTopK(logits, k_exp, c.expert_weights_scale, top_idx[0..k_exp], top_val[0..k_exp]);

            // 5. Evaluate selected top-K experts for token t
            const tok_moe_out = s.moe_out[t * dim ..][0..dim];

            for (0..k_exp) |j| {
                const exp_idx = top_idx[j];
                const weight = top_val[j];
                const exp = &moe.experts[exp_idx];

                s.xn.prepare(quant.actKind(exp.w_gate.typ), s.pool);

                const rc = rowsChunk(ffn_dim, s.pool.nth);
                var fj: FfnJob = .{
                    .gate = &exp.w_gate,
                    .up = &exp.w_up,
                    .act = &s.xn,
                    .out = s.hb.f,
                    .ntok = 1,
                    .tok_offset = t,
                    .rows_chunk = rc,
                    .gelu = c.gelu,
                    .chunks = .init((ffn_dim + rc - 1) / rc, 1),
                };
                s.pool.run(&fj, ffnWorker);
                s.hb.reset(1);

                // Down projection into s.tmp
                s.matmul(&.{.{ .w = &exp.w_down, .out = s.tmp.ptr, .stride = dim, .accumulate = false }}, &s.hb);

                // Accumulate scaled expert output: tok_moe_out += weight * s.tmp
                const w_vec: quant.V8f = @splat(weight);
                var d: usize = 0;
                while (d + 8 <= dim) : (d += 8) {
                    const tmp_v: quant.V8f = s.tmp[d..][0..8].*;
                    const out_v: quant.V8f = tok_moe_out[d..][0..8].*;
                    tok_moe_out[d..][0..8].* = @mulAdd(quant.V8f, w_vec, tmp_v, out_v);
                }
                while (d < dim) : (d += 1) {
                    tok_moe_out[d] += weight * s.tmp[d];
                }
            }

            // 6. Optional shared expert
            if (moe.shared) |*sh| {
                s.xn.prepare(quant.actKind(sh.w_gate.typ), s.pool);
                const rc = rowsChunk(c.ffn_dim, s.pool.nth);
                var fj: FfnJob = .{
                    .gate = &sh.w_gate,
                    .up = &sh.w_up,
                    .act = &s.xn,
                    .out = s.hb.f,
                    .ntok = 1,
                    .tok_offset = t,
                    .rows_chunk = rc,
                    .gelu = c.gelu,
                    .chunks = .init((c.ffn_dim + rc - 1) / rc, 1),
                };
                s.pool.run(&fj, ffnWorker);
                s.hb.reset(1);

                s.matmul(&.{.{ .w = &sh.w_down, .out = s.tmp.ptr, .stride = dim, .accumulate = false }}, &s.hb);

                var d: usize = 0;
                while (d + 8 <= dim) : (d += 8) {
                    const tmp_v: quant.V8f = s.tmp[d..][0..8].*;
                    const out_v: quant.V8f = tok_moe_out[d..][0..8].*;
                    tok_moe_out[d..][0..8].* = tmp_v + out_v;
                }
                while (d < dim) : (d += 1) {
                    tok_moe_out[d] += s.tmp[d];
                }
            }
        }

        // 7. Add MoE output into residual stream x
        for (0..nt) |t| {
            const x_tok = s.x[t * dim ..][0..dim];
            const moe_tok = s.moe_out[t * dim ..][0..dim];
            if (post_ffn_norm) |w| {
                rmsnorm(s.tmp[0..dim], moe_tok, w, c.eps);
                var d: usize = 0;
                while (d + 8 <= dim) : (d += 8) {
                    const tv: quant.V8f = s.tmp[d..][0..8].*;
                    const xv: quant.V8f = x_tok[d..][0..8].*;
                    x_tok[d..][0..8].* = xv + tv;
                }
                while (d < dim) : (d += 1) x_tok[d] += s.tmp[d];
            } else {
                var d: usize = 0;
                while (d + 8 <= dim) : (d += 8) {
                    const mv: quant.V8f = moe_tok[d..][0..8].*;
                    const xv: quant.V8f = x_tok[d..][0..8].*;
                    x_tok[d..][0..8].* = xv + mv;
                }
                while (d < dim) : (d += 1) x_tok[d] += moe_tok[d];
            }
        }
    }

    fn forwardBatch(s: *State, tokens: []const u32, n_logits: usize) void {
        const m = s.model;
        const c = &m.cfg;
        const nt = tokens.len;
        const pos0 = s.pos;
        for (tokens, 0..) |tok, t| {
            const xt = s.x[t * c.dim ..][0..c.dim];
            quant.dequantRow(m.embd.typ, m.embd.row(tok), xt);
            if (c.embd_scale != 1) for (xt) |*v| {
                v.* *= c.embd_scale;
            };
        }
        s.ropeTables(pos0, nt);

        // Gemma 4 Per-Layer Embeddings (PLE) input preparation
        if (m.tok_embd_per_layer) |*ple_embd| {
            const total_ple_dim = c.n_layers * c.n_embd_per_layer;
            const ple_scale = @sqrt(@as(f32, @floatFromInt(c.n_embd_per_layer)));
            for (tokens, 0..) |tok, t| {
                const row_slice = s.inp_per_layer[t * total_ple_dim ..][0..total_ple_dim];
                quant.dequantRow(ple_embd.typ, ple_embd.row(tok), row_slice);
                for (row_slice) |*v| v.* *= ple_scale;
            }

            if (m.per_layer_model_proj) |*proj| {
                s.xn.reset(nt);
                for (0..nt) |t| @memcpy(s.xn.f[t * c.dim ..][0..c.dim], s.x[t * c.dim ..][0..c.dim]);
                s.matmul(&.{.{ .w = proj, .out = s.ple_proj.ptr, .stride = total_ple_dim }}, &s.xn);

                const proj_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.dim)));
                for (s.ple_proj[0 .. nt * total_ple_dim]) |*v| v.* *= proj_scale;

                const inv_sqrt2: f32 = 1.0 / @sqrt(2.0);
                for (0..nt) |t| {
                    for (0..c.n_layers) |il| {
                        const prj_slice = s.ple_proj[t * total_ple_dim + il * c.n_embd_per_layer ..][0..c.n_embd_per_layer];
                        if (m.per_layer_proj_norm) |w| {
                            rmsnorm(prj_slice, prj_slice, w, c.eps);
                        }
                        const inp_slice = s.inp_per_layer[t * total_ple_dim + il * c.n_embd_per_layer ..][0..c.n_embd_per_layer];
                        for (inp_slice, prj_slice) |*inp, prj| {
                            inp.* = (inp.* + prj) * inv_sqrt2;
                        }
                    }
                }
            }
        }

        for (m.layers, 0..) |*L, il| {
            const attn_kind = switch (L.mixer) {
                .attention => |*A| quant.actKind(A.wq.typ),
                .shortconv => |*C| quant.actKind(C.in_proj.typ),
            };
            s.fusedNorm(nt, s.x.ptr, L.attn_norm, attn_kind);
            // Without a post-norm the mixer output accumulates straight into x.
            const mix_direct = L.post_mix_norm == null;
            const mix_dst = if (mix_direct) s.x.ptr else s.tmp.ptr;
            switch (L.mixer) {
                .attention => |*A| s.attention(A, nt, pos0, mix_dst, mix_direct),
                .shortconv => |*C| s.shortConv(C, nt, mix_dst, mix_direct),
            }
            if (L.post_mix_norm) |w| s.addNormed(w, nt);

            switch (L.ffn) {
                .dense => |*d| {
                    const ffn_kind = quant.actKind(d.w_gate.typ);
                    s.fusedNorm(nt, s.x.ptr, L.ffn_norm, ffn_kind);
                    const l_ffn = d.w_gate.rows;
                    const rc = rowsChunk(l_ffn, s.pool.nth);
                    var fj: FfnJob = .{ .gate = &d.w_gate, .up = &d.w_up, .act = &s.xn, .out = s.hb.f, .ntok = nt, .rows_chunk = rc, .gelu = c.gelu, .chunks = .init((l_ffn + rc - 1) / rc, 1) };
                    s.pool.run(&fj, ffnWorker);
                    s.hb.resetDim(l_ffn, nt);
                    const ffn_direct = L.post_ffn_norm == null;
                    s.matmul(&.{.{ .w = &d.w_down, .out = if (ffn_direct) s.x.ptr else s.tmp.ptr, .stride = c.dim, .accumulate = ffn_direct }}, &s.hb);
                    if (L.post_ffn_norm) |w| s.addNormed(w, nt);
                },
                .moe => |*moe| {
                    s.forwardMoE(moe, L.ffn_norm, L.post_ffn_norm, nt);
                },
            }

            // PLE layer projection
            if (L.inp_gate) |*gate| {
                const total_ple_dim = c.n_layers * c.n_embd_per_layer;
                s.xn.reset(nt);
                for (0..nt) |t| @memcpy(s.xn.f[t * c.dim ..][0..c.dim], s.x[t * c.dim ..][0..c.dim]);
                s.matmul(&.{.{ .w = gate, .out = s.ple_buf.ptr, .stride = c.n_embd_per_layer }}, &s.xn);

                for (0..nt) |t| {
                    const g_slice = s.ple_buf[t * c.n_embd_per_layer ..][0..c.n_embd_per_layer];
                    const inp_slice = s.inp_per_layer[t * total_ple_dim + il * c.n_embd_per_layer ..][0..c.n_embd_per_layer];
                    gelu_mul(g_slice, g_slice, inp_slice);
                }

                if (L.proj) |*proj| {
                    s.ple_act.reset(nt);
                    for (0..nt) |t| @memcpy(s.ple_act.f[t * c.n_embd_per_layer ..][0..c.n_embd_per_layer], s.ple_buf[t * c.n_embd_per_layer ..][0..c.n_embd_per_layer]);
                    s.matmul(&.{.{ .w = proj, .out = s.tmp.ptr, .stride = c.dim }}, &s.ple_act);

                    if (L.per_layer_post_norm) |w| {
                        for (0..nt) |t| {
                            const p_tok = s.tmp[t * c.dim ..][0..c.dim];
                            rmsnorm(p_tok, p_tok, w, c.eps);
                        }
                    }

                    for (0..nt) |t| {
                        const x_tok = s.x[t * c.dim ..][0..c.dim];
                        const p_tok = s.tmp[t * c.dim ..][0..c.dim];
                        var d: usize = 0;
                        while (d + 8 <= c.dim) : (d += 8) {
                            const xv: V8f = x_tok[d..][0..8].*;
                            const pv: V8f = p_tok[d..][0..8].*;
                            x_tok[d..][0..8].* = xv + pv;
                        }
                        while (d < c.dim) : (d += 1) x_tok[d] += p_tok[d];
                    }
                }
            }

            if (L.layer_output_scale) |scale| {
                for (0..nt) |t| {
                    const x_tok = s.x[t * c.dim ..][0..c.dim];
                    const sv: V8f = @splat(scale);
                    var d: usize = 0;
                    while (d + 8 <= c.dim) : (d += 8) {
                        const xv: V8f = x_tok[d..][0..8].*;
                        x_tok[d..][0..8].* = xv * sv;
                    }
                    while (d < c.dim) : (d += 1) x_tok[d] *= scale;
                }
            }
        }
        s.pos += nt;

        if (n_logits > 0) {
            std.debug.assert(n_logits <= tokens.len and n_logits <= max_spec + 1);
            const start = tokens.len - n_logits;
            const out_kind = quant.actKind(m.output.typ);
            s.fusedNorm(n_logits, s.x.ptr + start * c.dim, m.out_norm, out_kind);
            if (m.output_coarse != null) {
                for (0..n_logits) |t| s.outputCoarseToFine(t);
            } else {
                s.matmul(&.{.{ .w = &m.output, .out = s.logits.ptr, .stride = c.vocab }}, &s.xn);
            }
            if (c.final_logit_softcapping > 0) {
                const cap = c.final_logit_softcapping;
                const cap_v: V8f = @splat(cap);
                const inv_cap_v: V8f = @splat(1.0 / cap);
                for (0..n_logits) |t| {
                    const l_tok = s.logits[t * c.vocab ..][0..c.vocab];
                    var i: usize = 0;
                    while (i + 32 <= l_tok.len) : (i += 32) {
                        inline for (0..4) |unroll| {
                            const offset = i + unroll * 8;
                            const v: V8f = l_tok[offset..][0..8].*;
                            const z = v * inv_cap_v;
                            const z_clamped = @min(@max(z, @as(V8f, @splat(-9.0))), @as(V8f, @splat(9.0)));
                            const e2z = @exp(z_clamped * @as(V8f, @splat(2.0)));
                            const tanh_z = (e2z - @as(V8f, @splat(1.0))) / (e2z + @as(V8f, @splat(1.0)));
                            l_tok[offset..][0..8].* = cap_v * tanh_z;
                        }
                    }
                    while (i < l_tok.len) : (i += 1) {
                        l_tok[i] = cap * std.math.tanh(l_tok[i] / cap);
                    }
                }
            }
        }
    }

    /// Append tokens to the context and return logits for the last `n_logits` tokens.
    pub fn forwardN(s: *State, tokens: []const u32, n_logits: usize) ![]f32 {
        if (s.pos + tokens.len > s.ctx) return error.ContextFull;
        if (n_logits > max_spec + 1 or n_logits > tokens.len) return error.TooManyLogits;
        if (tokens.len <= s.bmax) {
            s.forwardBatch(tokens, n_logits);
            return s.logits[0 .. n_logits * s.model.cfg.vocab];
        }
        const no_logits_len = tokens.len - n_logits;
        var i: usize = 0;
        while (i < no_logits_len) {
            const n = @min(s.bmax, no_logits_len - i);
            s.forwardBatch(tokens[i..][0..n], 0);
            i += n;
        }
        if (n_logits > 0) {
            s.forwardBatch(tokens[no_logits_len..], n_logits);
        }
        return s.logits[0 .. n_logits * s.model.cfg.vocab];
    }

    /// Append tokens to the context and return logits of the last one.
    pub fn forward(s: *State, tokens: []const u32) ![]f32 {
        return s.forwardN(tokens, 1);
    }
};

