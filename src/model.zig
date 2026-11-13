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

pub const Arch = enum { llama, qwen2, qwen3, lfm3 };

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
    /// Sliding-window attention layers (0 = none): window size and RoPE base.
    swa_window: usize = 0,
    rope_base_swa: f32 = 10000,
    /// Short-convolution layers (LFM): kernel length, 0 = none.
    conv_kernel: usize = 0,
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

const Attention = struct {
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
};

/// LFM gated short convolution: y = C * conv(B * x), all from one in_proj.
const ShortConv = struct {
    in_proj: Tensor,
    out_proj: Tensor,
    /// Depthwise causal kernel, transposed to [tap][channel].
    kernel: []const f32,
    /// Index of this layer's convolution state.
    slot: usize,
};

const Layer = struct {
    attn_norm: []const f32,
    ffn_norm: []const f32,
    /// Sandwich norms applied to the mixer / FFN output before the residual add.
    post_mix_norm: ?[]const f32 = null,
    post_ffn_norm: ?[]const f32 = null,
    mixer: union(enum) { attention: Attention, shortconv: ShortConv },
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
};

pub const LoadOptions = struct {
    /// Interleave weights for the x8 kernels (otherwise plain per-row kernels).
    repack: bool = true,
};

pub const Model = struct {
    file: gguf.File,
    cfg: Config,
    /// Token embeddings, read row-wise straight from the mapped file.
    embd: Tensor,
    output: Tensor,
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
        try self.materialize(gpa, pool, opts.repack);
        return self;
    }

    pub fn deinit(self: *Model, gpa: Allocator) void {
        for (self.owned.items) |b| gpa.free(b);
        self.owned.deinit(gpa);
        gpa.free(self.layers);
        gpa.free(self.inv_freq);
        gpa.free(self.inv_freq_swa);
        if (self.weights) |w| posix.munmap(w);
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
            try list.appendSlice(gpa, &.{ &L.w_gate, &L.w_up, &L.w_down });
        }
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

    /// Move every matmul weight into one huge-page region, repacking when possible.
    fn materialize(self: *Model, gpa: Allocator, pool: *Pool, use_repack: bool) !void {
        const ts = try self.matmulTensors(gpa);
        defer gpa.free(ts);
        var total: usize = 0;
        for (ts) |t| total += t.materializedBytes(use_repack);
        const len = std.mem.alignForward(usize, total, 2 << 20);
        const region = try posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        self.weights = region;
        posix.madvise(region.ptr, len, posix.MADV.HUGEPAGE) catch {};
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
        const arch = std.meta.stringToEnum(Arch, arch_s) orelse {
            std.log.err("unsupported architecture '{s}' (supported: llama, qwen2, qwen3, lfm3)", .{arch_s});
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
        c.ffn_dim = try req(f, self.key(&kb, "feed_forward_length"));
        c.head_dim = @intCast(f.getUint(self.key(&kb, "attention.key_length")) orelse c.dim / c.n_heads);
        c.q_dim = c.n_heads * c.head_dim;
        c.kv_dim = c.n_kv_heads * c.head_dim;
        c.eps = @floatCast(f.getFloat(self.key(&kb, "attention.layer_norm_rms_epsilon")) orelse 1e-5);
        c.rope_base = @floatCast(f.getFloat(self.key(&kb, "rope.freq_base")) orelse 10000);
        c.n_rot = @intCast(f.getUint(self.key(&kb, "rope.dimension_count")) orelse c.head_dim);
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
        c.rope_base_swa = @floatCast(f.getFloat(self.key(&kb, "rope.freq_base_swa")) orelse c.rope_base);
        c.conv_kernel = @intCast(f.getUint(self.key(&kb, "shortconv.l_cache")) orelse 0);
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
        const a = self.file.getArray(self.key(&kb, "attention.sliding_window_pattern")) orelse return false;
        return a.elem == .bool and i < a.len and a.item(u8, i) != 0;
    }

    fn tensor(self: *Model, name: []const u8, cols: usize, rows: ?usize) !Tensor {
        const ti = self.file.tensors.get(name) orelse {
            std.log.err("missing tensor {s}", .{name});
            return error.MissingTensor;
        };
        if (!quant.isSupported(ti.typ)) {
            std.log.err("tensor {s}: unsupported type {d} (supported: F32 F16 BF16 Q4_0 Q8_0 Q4_K Q5_K Q6_K)", .{ name, @backingInt(ti.typ) });
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
        self.out_norm = try self.vector(gpa, "output_norm.weight", c.dim);
        self.layers = try gpa.alloc(Layer, c.n_layers);
        var nb: [96]u8 = undefined;
        for (self.layers, 0..) |*L, i| {
            const n = struct {
                fn f(buf: []u8, l: usize, comptime s: []const u8) []const u8 {
                    return std.fmt.bufPrint(buf, "blk.{d}." ++ s, .{l}) catch unreachable;
                }
            }.f;
            L.* = .{
                .attn_norm = try self.vector(gpa, n(&nb, i, "attn_norm.weight"), c.dim),
                .ffn_norm = try self.vector(gpa, n(&nb, i, "ffn_norm.weight"), c.dim),
                .post_mix_norm = try self.optVector(gpa, n(&nb, i, "post_attention_norm.weight"), c.dim),
                .post_ffn_norm = try self.optVector(gpa, n(&nb, i, "post_ffw_norm.weight"), c.dim),
                .mixer = undefined,
                .w_gate = try self.tensor(n(&nb, i, "ffn_gate.weight"), c.dim, c.ffn_dim),
                .w_up = try self.tensor(n(&nb, i, "ffn_up.weight"), c.dim, c.ffn_dim),
                .w_down = try self.tensor(n(&nb, i, "ffn_down.weight"), c.ffn_dim, c.dim),
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
                const kv_dim = kv * c.head_dim;
                L.mixer = .{ .attention = .{
                    .wq = try self.tensor(n(&nb, i, "attn_q.weight"), c.dim, c.q_dim),
                    .wk = try self.tensor(n(&nb, i, "attn_k.weight"), c.dim, kv_dim),
                    .wv = try self.tensor(n(&nb, i, "attn_v.weight"), c.dim, kv_dim),
                    .wo = try self.tensor(n(&nb, i, "attn_output.weight"), c.q_dim, c.dim),
                    .gate = if (self.file.tensors.contains(n(&nb, i, "attn_gate.weight"))) try self.tensor(n(&nb, i, "attn_gate.weight"), c.dim, c.n_heads) else null,
                    .bq = try self.optVector(gpa, n(&nb, i, "attn_q.bias"), c.q_dim),
                    .bk = try self.optVector(gpa, n(&nb, i, "attn_k.bias"), kv_dim),
                    .bv = try self.optVector(gpa, n(&nb, i, "attn_v.bias"), kv_dim),
                    .q_norm = try self.optVector(gpa, n(&nb, i, "attn_q_norm.weight"), c.head_dim),
                    .k_norm = try self.optVector(gpa, n(&nb, i, "attn_k_norm.weight"), c.head_dim),
                    .n_kv_heads = kv,
                    .swa = self.layerIsSwa(i),
                    .kv_slot = self.n_attn_layers,
                } };
                self.n_attn_layers += 1;
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

    fn ropeFreqs(c: *const Config, base: f32, factors: ?[]const f32, out: []f32) void {
        for (out, 0..) |*v, i| {
            const e = -2.0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(c.n_rot));
            var w = std.math.pow(f64, base, e) * c.rope_scale;
            if (factors) |fr| w /= fr[i];
            v.* = @floatCast(w);
        }
    }

    fn buildRope(self: *Model, gpa: Allocator) !void {
        const c = &self.cfg;
        const half = c.n_rot / 2;
        self.inv_freq = try gpa.alloc(f32, half);
        // rope_freqs factors belong to the full-attention layers only.
        ropeFreqs(c, c.rope_base, try self.optVector(gpa, "rope_freqs.weight", half), self.inv_freq);
        if (c.swa_window > 0) {
            self.inv_freq_swa = try gpa.alloc(f32, half);
            ropeFreqs(c, c.rope_base_swa, null, self.inv_freq_swa);
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
const tok_group = 4;

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
    var ts = t0;
    while (ts < t1) : (ts += tok_tile) {
        const te = @min(ts + tok_tile, t1);
        var r = r0;
        while (r < r1) : (r += 8) {
            const g = w.group(r);
            var tk = ts;
            while (tk < te) {
                switch (@min(te - tk, tok_group)) {
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
}

fn rowsDispatch(acc: bool, w: *const Tensor, r0: usize, r1: usize, act: *const Act, t0: usize, t1: usize, out: [*]f32, ostride: usize) void {
    if (w.x8) {
        switch (w.typ) {
            inline .q4_0, .q8_0, .q4_k, .q5_k, .q6_k => |t| {
                if (acc) groupsDot(t, true, w, r0, r1, act, t0, t1, out, ostride) else groupsDot(t, false, w, r0, r1, act, t0, t1, out, ostride);
            },
            else => unreachable,
        }
        return;
    }
    switch (w.typ) {
        inline .f32, .f16, .bf16, .q4_0, .q8_0, .q4_k, .q5_k, .q6_k => |t| {
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
            rowsDispatch(o.accumulate, o.w, r0, r1, job.act, 0, job.ntok, o.out + r0, o.stride);
        }
    }
}

const FfnJob = struct {
    gate: *const Tensor,
    up: *const Tensor,
    act: *const Act,
    out: []f32,
    ntok: usize,
    rows_chunk: usize,
    chunks: Chunks,
};

fn ffnWorker(job: *FfnJob, _: usize, _: usize) void {
    var g: [max_rows_chunk * tok_tile]f32 = undefined;
    var u: [max_rows_chunk * tok_tile]f32 = undefined;
    const ffn = job.gate.rows;
    while (job.chunks.grab()) |range| {
        const r0 = range[0] * job.rows_chunk;
        const r1 = @min(range[1] * job.rows_chunk, ffn);
        const nr = r1 - r0;
        var ts: usize = 0;
        while (ts < job.ntok) : (ts += tok_tile) {
            const te = @min(ts + tok_tile, job.ntok);
            rowsDispatch(false, job.gate, r0, r1, job.act, ts, te, &g, nr);
            rowsDispatch(false, job.up, r0, r1, job.act, ts, te, &u, nr);
            for (ts..te) |tk| {
                const gi = g[(tk - ts) * nr ..][0..nr];
                const ui = u[(tk - ts) * nr ..][0..nr];
                silu_mul(job.out[tk * ffn + r0 ..][0..nr], gi, ui);
            }
        }
    }
}

const V8f = @Vector(8, f32);

/// Cephes-style expf on 8 lanes (rel. error ~1e-7).
inline fn expV(x_in: V8f) V8f {
    var x = @min(@max(x_in, @as(V8f, @splat(-87.3))), @as(V8f, @splat(88.3)));
    const fx = @floor(x * @as(V8f, @splat(1.44269504088896341)) + @as(V8f, @splat(0.5)));
    x = x - fx * @as(V8f, @splat(0.693359375)) - fx * @as(V8f, @splat(-2.12194440e-4));
    var y: V8f = @splat(1.9875691500e-4);
    y = y * x + @as(V8f, @splat(1.3981999507e-3));
    y = y * x + @as(V8f, @splat(8.3334519073e-3));
    y = y * x + @as(V8f, @splat(4.1665795894e-2));
    y = y * x + @as(V8f, @splat(1.6666665459e-1));
    y = y * x + @as(V8f, @splat(5.0000001201e-1));
    y = y * x * x + x + @as(V8f, @splat(1.0));
    const n: @Vector(8, i32) = @intFromFloat(fx);
    const p2: V8f = @bitCast((n + @as(@Vector(8, i32), @splat(127))) << @splat(23));
    return y * p2;
}

fn silu_mul(out: []f32, g: []const f32, u: []const f32) void {
    var i: usize = 0;
    while (i + 8 <= out.len) : (i += 8) {
        const gv: V8f = g[i..][0..8].*;
        const uv: V8f = u[i..][0..8].*;
        out[i..][0..8].* = gv / (@as(V8f, @splat(1)) + expV(-gv)) * uv;
    }
    while (i < out.len) : (i += 1) out[i] = g[i] / (1 + @exp(-g[i])) * u[i];
}

fn rmsnorm(out: []f32, x: []const f32, w: []const f32, eps: f32) void {
    var ss: V8f = @splat(0);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) {
        const v: V8f = x[i..][0..8].*;
        ss += v * v;
    }
    var s = @reduce(.Add, ss);
    while (i < x.len) : (i += 1) s += x[i] * x[i];
    const scale = 1.0 / @sqrt(s / @as(f32, @floatFromInt(x.len)) + eps);
    for (out, x, w) |*o, v, g| o.* = v * scale * g;
}

// ---------------------------------------------------------------------------
// Attention
//
// K is cached transposed in tiles of 16 positions ([head_dim][16] f16 per
// tile) so a score block is a tiny GEMM: each K tile is converted to f32 once
// and reused by up to 4 query vectors (GQA heads and/or prompt tokens) that
// share the KV head. V stays row-major. Softmax is computed online per tile
// (flash-attention style), so no score buffers scale with context length.

pub const kv_tile = 16;
const max_q = 4;
const max_head_dim = 256;

const V16f = @Vector(16, f32);

/// Online-softmax attention of NQ queries against one KV head; query i sees
/// positions [lo[i], np[i]). Writes the normalized outputs to `out`.
fn attendBlock(comptime NQ: usize, hd: usize, kt: [*]const f16, vr: [*]const f16, q: [NQ][*]const f32, lo: [NQ]usize, np: [NQ]usize, out: [NQ][*]f32, scale: f32) void {
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

    for (minlo / kv_tile..ntiles) |j| {
        const tile = kt + j * hd * kv_tile;
        var s: [NQ]V16f = @splat(@splat(0));
        for (0..hd) |d| {
            const kf: V16f = @floatCast(@as(@Vector(16, f16), tile[d * kv_tile ..][0..kv_tile].*));
            inline for (0..NQ) |i| s[i] = @mulAdd(V16f, @splat(q[i][d]), kf, s[i]);
        }
        // Mask, then fold the tile into the running softmax of each query.
        var p: [NQ][kv_tile]f32 = undefined;
        const base = j * kv_tile;
        inline for (0..NQ) |i| {
            var sv = s[i] * @as(V16f, @splat(scale));
            if (base < lo[i] or base + kv_tile > np[i]) {
                const lane = std.simd.iota(u32, kv_tile) + @as(@Vector(kv_tile, u32), @splat(@intCast(base)));
                const visible = (lane < @as(@Vector(kv_tile, u32), @splat(@intCast(np[i])))) & (lane >= @as(@Vector(kv_tile, u32), @splat(@intCast(lo[i]))));
                sv = @select(f32, visible, sv, @as(V16f, @splat(-std.math.inf(f32))));
            }
            const tmax = @reduce(.Max, sv);
            if (tmax == -std.math.inf(f32)) {
                p[i] = @splat(0);
            } else {
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
                    while (d < hd) : (d += 8) o[i][d..][0..8].* = @as(V8f, o[i][d..][0..8].*) * cv;
                }
                m[i] = m_new;
            }
        }
        // o += p . V for this tile, 8 output dims at a time kept in registers.
        const nvalid = @min(kv_tile, maxnp - base);
        const vt = vr + base * hd;
        var d: usize = 0;
        while (d < hd) : (d += 8) {
            var acc: [NQ]V8f = undefined;
            inline for (0..NQ) |i| acc[i] = o[i][d..][0..8].*;
            for (0..nvalid) |n| {
                const vv: V8f = @floatCast(@as(@Vector(8, f16), vt[n * hd + d ..][0..8].*));
                inline for (0..NQ) |i| acc[i] = @mulAdd(V8f, @splat(p[i][n]), vv, acc[i]);
            }
            inline for (0..NQ) |i| o[i][d..][0..8].* = acc[i];
        }
    }
    inline for (0..NQ) |i| {
        const inv: V8f = @splat(if (l[i] > 0) 1.0 / l[i] else 0);
        var d: usize = 0;
        while (d < hd) : (d += 8) out[i][d..][0..8].* = @as(V8f, o[i][d..][0..8].*) * inv;
    }
}

const AttnJob = struct {
    st: *State,
    slot: usize,
    n_kv: usize,
    /// Sliding window length, 0 = full causal attention.
    window: usize,
    pos0: usize,
    ntok: usize,
    /// Query blocks per KV head.
    qblocks: usize,
    chunks: Chunks,
};

fn attnWorker(job: *AttnJob, _: usize, _: usize) void {
    const st = job.st;
    const c = &st.model.cfg;
    const hd = c.head_dim;
    const g = c.n_heads / job.n_kv;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const nq_total = job.ntok * g;
    while (job.chunks.grab()) |range| for (range[0]..range[1]) |item| {
        const kh = item / job.qblocks;
        const qb = item % job.qblocks;
        const first = qb * max_q;
        const nq = @min(max_q, nq_total - first);
        const kt = st.k_cache.ptr + st.cacheIndex(job.slot, kh);
        const vr = st.v_cache.ptr + st.cacheIndex(job.slot, kh);
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
                    qp[i] = st.q.ptr + t * c.q_dim + h * hd;
                    np[i] = job.pos0 + t + 1;
                    lo[i] = if (job.window > 0 and np[i] > job.window) np[i] - job.window else 0;
                    op[i] = st.att.f.ptr + t * c.q_dim + h * hd;
                }
                attendBlock(n, hd, kt, vr, qp, lo, np, op, scale);
            },
            else => unreachable,
        }
    };
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
    /// Per (attention layer, kv head): K as [ctx/16 tiles][head_dim][16], V as [ctx][head_dim].
    k_cache: []f16,
    v_cache: []f16,
    /// ctx rounded up to whole K tiles.
    ctx_pad: usize,
    /// Mixer / FFN output awaiting its post-norm (sandwich-norm models).
    tmp: []f32,
    /// Per-head attention gate logits [bmax][n_heads].
    gate: []f32,
    /// Short conv: in_proj output [bmax][3*dim], input history
    /// [(kernel-1) + bmax][dim], and the last kernel-1 inputs of every conv layer.
    conv_in: []f32,
    conv_hist: []f32,
    conv_state: []f32,

    pub fn init(gpa: Allocator, model: *const Model, pool: *Pool, ctx: usize, bmax: usize) !State {
        const c = &model.cfg;
        if (c.head_dim % 8 != 0 or c.head_dim > max_head_dim) return error.UnsupportedHeadDim;
        var s: State = undefined;
        s.gpa = gpa;
        s.model = model;
        s.pool = pool;
        s.ctx = ctx;
        s.bmax = bmax;
        s.pos = 0;
        s.ctx_pad = std.mem.alignForward(usize, ctx, kv_tile);
        const hist = if (c.conv_kernel > 0) c.conv_kernel - 1 else 0;
        const has_conv = model.n_conv_layers > 0;
        s.xn = try Act.init(gpa, c.dim, bmax);
        errdefer s.xn.deinit(gpa);
        s.att = try Act.init(gpa, c.q_dim, bmax);
        errdefer s.att.deinit(gpa);
        s.hb = try Act.init(gpa, c.ffn_dim, bmax);
        errdefer s.hb.deinit(gpa);
        // Plain f32 buffers: one allocation each, freed together on failure.
        const bufs = .{
            .{ "x", bmax * c.dim },
            .{ "q", bmax * c.q_dim },
            .{ "k", bmax * c.kv_dim },
            .{ "v", bmax * c.kv_dim },
            .{ "logits", c.vocab },
            .{ "rope_cs", bmax * c.n_rot },
            .{ "rope_cs_swa", if (model.inv_freq_swa.len > 0) bmax * c.n_rot else 0 },
            .{ "tmp", bmax * c.dim },
            .{ "gate", bmax * c.n_heads },
            .{ "conv_in", if (has_conv) bmax * 3 * c.dim else 0 },
            .{ "conv_hist", if (has_conv) (hist + bmax) * c.dim else 0 },
            .{ "conv_state", model.n_conv_layers * hist * c.dim },
        };
        var done: usize = 0;
        errdefer inline for (bufs, 0..) |b, i| {
            if (i < done) gpa.free(@field(s, b[0]));
        };
        inline for (bufs) |b| {
            @field(s, b[0]) = try gpa.alloc(f32, b[1]);
            done += 1;
        }
        @memset(s.conv_state, 0);
        const cache_len = model.n_attn_layers * c.n_kv_heads * s.ctx_pad * c.head_dim;
        s.k_cache = try gpa.alloc(f16, cache_len);
        errdefer gpa.free(s.k_cache);
        s.v_cache = try gpa.alloc(f16, cache_len);
        return s;
    }

    pub fn deinit(s: *State) void {
        const gpa = s.gpa;
        s.xn.deinit(gpa);
        s.att.deinit(gpa);
        s.hb.deinit(gpa);
        inline for (.{ "x", "q", "k", "v", "logits", "rope_cs", "rope_cs_swa", "tmp", "gate", "conv_in", "conv_hist", "conv_state" }) |name| gpa.free(@field(s, name));
        gpa.free(s.k_cache);
        gpa.free(s.v_cache);
    }

    /// Forget the context (KV cache positions and convolution state).
    pub fn reset(s: *State) void {
        s.pos = 0;
        @memset(s.conv_state, 0);
    }

    /// Start of one (attention layer, kv head) region in k_cache / v_cache.
    inline fn cacheIndex(s: *const State, slot: usize, kh: usize) usize {
        const c = &s.model.cfg;
        return (slot * c.n_kv_heads + kh) * s.ctx_pad * c.head_dim;
    }

    fn matmul(s: *State, outs: []const MatOut, act: *Act) void {
        std.debug.assert(outs.len <= max_outs);
        var rows: usize = 0;
        for (outs) |o| {
            act.prepare(quant.actKind(o.w.typ), s.pool);
            rows += o.w.rows;
        }
        const rc = rowsChunk(rows, s.pool.nth);
        var job: MatJob = .{ .outs = outs, .act = act, .ntok = act.ntok, .rows_chunk = rc, .first = undefined, .chunks = undefined };
        job.first[0] = 0;
        for (outs, 0..) |o, i| job.first[i + 1] = job.first[i] + (o.w.rows + rc - 1) / rc;
        job.chunks = .init(job.first[outs.len], 1);
        s.pool.run(&job, matWorker);
    }

    /// Rows per work item: a multiple of 8 (x8 groups), ~8 items per thread.
    fn rowsChunk(rows: usize, nth: usize) usize {
        return std.math.clamp(rows / (nth * 8) / 8 * 8, 8, max_rows_chunk);
    }

    fn ropeTables(s: *State, pos0: usize, ntok: usize) void {
        const m = s.model;
        const n_rot = m.cfg.n_rot;
        const half = n_rot / 2;
        for (0..ntok) |t| {
            const p: f32 = @floatFromInt(pos0 + t);
            for ([_][]const f32{ m.inv_freq, m.inv_freq_swa }, [_][]f32{ s.rope_cs, s.rope_cs_swa }) |freqs, table| {
                if (freqs.len == 0) continue;
                const cs = table[t * n_rot ..];
                for (freqs, 0..) |f, i| {
                    const a = p * f;
                    cs[i] = @cos(a);
                    cs[half + i] = @sin(a);
                }
            }
        }
    }

    fn rope(s: *const State, v: []f32, t: usize, swa: bool) void {
        const c = &s.model.cfg;
        const half = c.n_rot / 2;
        const cs = (if (swa) s.rope_cs_swa else s.rope_cs)[t * c.n_rot ..];
        var h: usize = 0;
        while (h < v.len) : (h += c.head_dim) {
            const x = v[h..][0..c.head_dim];
            for (0..half) |i| {
                const ia = if (c.rope_neox) i else 2 * i;
                const ib = if (c.rope_neox) i + half else 2 * i + 1;
                const a = x[ia];
                const b = x[ib];
                x[ia] = a * cs[i] - b * cs[half + i];
                x[ib] = a * cs[half + i] + b * cs[i];
            }
        }
    }

    /// Attention mixer for `nt` tokens of normalized input in `xn`; the
    /// projected output is written to (or accumulated into) `dst`.
    fn attention(s: *State, A: *const Attention, nt: usize, pos0: usize, dst: [*]f32, accumulate: bool) void {
        const c = &s.model.cfg;
        const hd = c.head_dim;
        const kv_dim = A.n_kv_heads * hd;
        var outs: [max_outs]MatOut = .{
            .{ .w = &A.wq, .out = s.q.ptr, .stride = c.q_dim },
            .{ .w = &A.wk, .out = s.k.ptr, .stride = kv_dim },
            .{ .w = &A.wv, .out = s.v.ptr, .stride = kv_dim },
            undefined,
        };
        var n_outs: usize = 3;
        if (A.gate) |*g| {
            outs[3] = .{ .w = g, .out = s.gate.ptr, .stride = c.n_heads };
            n_outs = 4;
        }
        s.matmul(outs[0..n_outs], &s.xn);

        for (0..nt) |t| {
            const q = s.q[t * c.q_dim ..][0..c.q_dim];
            const k = s.k[t * kv_dim ..][0..kv_dim];
            const v = s.v[t * kv_dim ..][0..kv_dim];
            if (A.bq) |b| for (q, b) |*a, bb| {
                a.* += bb;
            };
            if (A.bk) |b| for (k, b) |*a, bb| {
                a.* += bb;
            };
            if (A.bv) |b| for (v, b) |*a, bb| {
                a.* += bb;
            };
            if (A.q_norm) |w| {
                var h: usize = 0;
                while (h < q.len) : (h += hd) rmsnorm(q[h..][0..hd], q[h..][0..hd], w, c.eps);
            }
            if (A.k_norm) |w| {
                var h: usize = 0;
                while (h < k.len) : (h += hd) rmsnorm(k[h..][0..hd], k[h..][0..hd], w, c.eps);
            }
            s.rope(q, t, A.swa);
            s.rope(k, t, A.swa);
            const p = pos0 + t;
            for (0..A.n_kv_heads) |kh| {
                const base = s.cacheIndex(A.kv_slot, kh);
                const kt = s.k_cache[base + (p / kv_tile) * hd * kv_tile + p % kv_tile ..];
                const vr = s.v_cache[base + p * hd ..][0..hd];
                for (0..hd) |d| {
                    kt[d * kv_tile] = @floatCast(k[kh * hd + d]);
                    vr[d] = @floatCast(v[kh * hd + d]);
                }
            }
        }

        const g = c.n_heads / A.n_kv_heads;
        const qblocks = (nt * g + max_q - 1) / max_q;
        var aj: AttnJob = .{
            .st = s,
            .slot = A.kv_slot,
            .n_kv = A.n_kv_heads,
            .window = if (A.swa) c.swa_window else 0,
            .pos0 = pos0,
            .ntok = nt,
            .qblocks = qblocks,
            .chunks = .init(A.n_kv_heads * qblocks, 1),
        };
        s.pool.run(&aj, attnWorker);

        if (A.gate != null) {
            for (0..nt) |t| for (0..c.n_heads) |h| {
                const gv = 1.0 / (1.0 + @exp(-s.gate[t * c.n_heads + h]));
                for (s.att.f[t * c.q_dim + h * hd ..][0..hd]) |*a| a.* *= gv;
            };
        }
        s.att.reset(nt);
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

    fn forwardBatch(s: *State, tokens: []const u32, want_logits: bool) void {
        const m = s.model;
        const c = &m.cfg;
        const nt = tokens.len;
        const pos0 = s.pos;
        for (tokens, 0..) |tok, t| quant.dequantRow(m.embd.typ, m.embd.row(tok), s.x[t * c.dim ..][0..c.dim]);
        s.ropeTables(pos0, nt);

        for (m.layers) |*L| {
            for (0..nt) |t| rmsnorm(s.xn.f[t * c.dim ..][0..c.dim], s.x[t * c.dim ..][0..c.dim], L.attn_norm, c.eps);
            s.xn.reset(nt);
            // Without a post-norm the mixer output accumulates straight into x.
            const mix_direct = L.post_mix_norm == null;
            const mix_dst = if (mix_direct) s.x.ptr else s.tmp.ptr;
            switch (L.mixer) {
                .attention => |*A| s.attention(A, nt, pos0, mix_dst, mix_direct),
                .shortconv => |*C| s.shortConv(C, nt, mix_dst, mix_direct),
            }
            if (L.post_mix_norm) |w| s.addNormed(w, nt);

            for (0..nt) |t| rmsnorm(s.xn.f[t * c.dim ..][0..c.dim], s.x[t * c.dim ..][0..c.dim], L.ffn_norm, c.eps);
            s.xn.reset(nt);
            s.xn.prepare(quant.actKind(L.w_gate.typ), s.pool);
            s.xn.prepare(quant.actKind(L.w_up.typ), s.pool);
            const rc = rowsChunk(c.ffn_dim, s.pool.nth);
            var fj: FfnJob = .{ .gate = &L.w_gate, .up = &L.w_up, .act = &s.xn, .out = s.hb.f, .ntok = nt, .rows_chunk = rc, .chunks = .init((c.ffn_dim + rc - 1) / rc, 1) };
            s.pool.run(&fj, ffnWorker);
            s.hb.reset(nt);
            const ffn_direct = L.post_ffn_norm == null;
            s.matmul(&.{.{ .w = &L.w_down, .out = if (ffn_direct) s.x.ptr else s.tmp.ptr, .stride = c.dim, .accumulate = ffn_direct }}, &s.hb);
            if (L.post_ffn_norm) |w| s.addNormed(w, nt);
        }
        s.pos += nt;

        if (want_logits) {
            rmsnorm(s.xn.f[0..c.dim], s.x[(nt - 1) * c.dim ..][0..c.dim], m.out_norm, c.eps);
            s.xn.reset(1);
            s.matmul(&.{.{ .w = &m.output, .out = s.logits.ptr, .stride = c.vocab }}, &s.xn);
        }
    }

    /// Append tokens to the context and return logits of the last one.
    pub fn forward(s: *State, tokens: []const u32) ![]f32 {
        if (s.pos + tokens.len > s.ctx) return error.ContextFull;
        var i: usize = 0;
        while (i < tokens.len) {
            const n = @min(s.bmax, tokens.len - i);
            s.forwardBatch(tokens[i..][0..n], i + n == tokens.len);
            i += n;
        }
        return s.logits;
    }
};

test "tiled online-softmax attention matches naive reference" {
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    inline for (.{ 64, 128 }) |hd| {
        const npos = 53; // not a multiple of the tile size
        const ntiles = (npos + kv_tile - 1) / kv_tile;
        var kt: [ntiles * hd * kv_tile]f16 = undefined;
        var vr: [ntiles * kv_tile * hd]f16 = undefined;
        var kf: [npos][hd]f32 = undefined;
        var vf: [npos][hd]f32 = undefined;
        for (&kt) |*x| x.* = std.math.nan(f16); // unwritten lanes must be masked
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
        // Query 0 uses a sliding window starting mid-tile.
        const lo = [3]usize{ 21, 0, 0 };
        const np = [3]usize{ npos, 17, 1 };
        var out: [3][hd]f32 = undefined;
        const scale: f32 = 0.125;
        attendBlock(3, hd, &kt, &vr, .{ &q[0], &q[1], &q[2] }, lo, np, .{ &out[0], &out[1], &out[2] }, scale);
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
