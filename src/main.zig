const std = @import("std");
const gguf = @import("gguf.zig");
const model_mod = @import("model.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const sampler_mod = @import("sampler.zig");
const pool_mod = @import("pool.zig");
const server_mod = @import("server.zig");
const gen_mod = @import("gen.zig");
const Timer = gen_mod.Timer;

const usage =
    \\usage: zinfer -m MODEL.gguf [options]
    \\
    \\  -p, --prompt TEXT     prompt text (special tokens like <|eot_id|> are parsed)
    \\  -f, --file PATH       read prompt from file
    \\  -n, --predict N       max tokens to generate (default 256)
    \\  --chat                interactive chat using the model's chat template
    \\  -s, --system TEXT     system message for --chat
    \\  --no-think            start assistant turns with an empty think block (Qwen3, LFM3)
    \\  -t, --threads N       worker threads (default: physical cores)
    \\  -c, --ctx N           context length (default: min(4096, trained))
    \\  -b, --batch N         prompt batch size (default 64)
    \\
    \\sampling (defaults: the model's general.sampling.* metadata, else the values shown)
    \\  --temp F              temperature, 0 = greedy (0.8)
    \\  --top-k N             0 = off (40)
    \\  --top-p F             (0.95)
    \\  --min-p F             (0.05)
    \\  --repeat-penalty F    penalize the last 64 tokens, 1 = off (1.0)
    \\  --seed N              RNG seed (time)
    \\
    \\speculative decoding:
    \\  --draft N             n-gram prompt lookup draft tokens, 0 = off (default 0, 3-5 recommended)
    \\  --draft-model PATH    small model as draft; --draft sets lookahead (default 3)
    \\                        (draft must share the target's tokenizer)
    \\
    \\  --server              run OpenAI / llama.cpp-compatible HTTP server
    \\  --port N              server port (default 8080; implies --server)
    \\  --no-prune            keep full lm_head matmul (for A/B comparison)
    \\  --no-repack           keep GGUF row layout (per-row kernels; for A/B comparison)
    \\  --cache f16|q8_0      K/V cache format; q8_0 halves cache RAM and
    \\                        attention bandwidth (f16)
    \\  --bench PP,TG         benchmark prompt processing of PP tokens and generation of TG tokens
    \\  -v, --verbose         print model info
    \\
;

/// Sampling options given on the command line; unset ones fall back to the
/// model's metadata, then to sampler defaults.
const SamplingArgs = struct {
    temp: ?f32 = null,
    top_k: ?usize = null,
    top_p: ?f32 = null,
    min_p: ?f32 = null,
    repeat_penalty: ?f32 = null,
    seed: ?u64 = null,
};

const Args = struct {
    model: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    prompt_file: ?[]const u8 = null,
    system: ?[]const u8 = null,
    n_predict: usize = 256,
    chat: bool = false,
    no_think: bool = false,
    server: bool = false,
    port: u16 = 8080,
    draft: usize = 0,
    draft_model: ?[]const u8 = null,
    threads: usize = 0,
    ctx: usize = 0,
    batch: usize = 64,
    sampling: SamplingArgs = .{},
    load: model_mod.LoadOptions = .{},
    cache_q8: bool = false,
    bench: ?[2]usize = null,
    verbose: bool = false,
};

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print(fmt ++ "\n", args);
    std.process.exit(1);
}

const ArgReader = struct {
    argv: []const [:0]const u8,
    i: usize = 1,

    fn is(r: *const ArgReader, comptime names: []const []const u8) bool {
        inline for (names) |n| if (std.mem.eql(u8, r.argv[r.i], n)) return true;
        return false;
    }

    fn str(r: *ArgReader) []const u8 {
        r.i += 1;
        if (r.i >= r.argv.len) fail("missing value for {s}", .{r.argv[r.i - 1]});
        return r.argv[r.i];
    }

    fn uint(r: *ArgReader) usize {
        const s = r.str();
        return std.fmt.parseInt(usize, s, 10) catch fail("bad number '{s}'", .{s});
    }

    fn float(r: *ArgReader) f32 {
        const s = r.str();
        return std.fmt.parseFloat(f32, s) catch fail("bad number '{s}'", .{s});
    }
};

fn parseArgs(argv: []const [:0]const u8) Args {
    var a: Args = .{};
    var r: ArgReader = .{ .argv = argv };
    while (r.i < argv.len) : (r.i += 1) {
        if (r.is(&.{ "-m", "--model" })) {
            a.model = r.str();
        } else if (r.is(&.{ "-p", "--prompt" })) {
            a.prompt = r.str();
        } else if (r.is(&.{ "-f", "--file" })) {
            a.prompt_file = r.str();
        } else if (r.is(&.{ "-s", "--system" })) {
            a.system = r.str();
        } else if (r.is(&.{ "-n", "--predict" })) {
            a.n_predict = r.uint();
        } else if (r.is(&.{ "-t", "--threads" })) {
            a.threads = r.uint();
        } else if (r.is(&.{ "-c", "--ctx" })) {
            a.ctx = r.uint();
        } else if (r.is(&.{ "-b", "--batch" })) {
            a.batch = @max(1, r.uint());
        } else if (r.is(&.{"--temp"})) {
            a.sampling.temp = r.float();
        } else if (r.is(&.{"--top-k"})) {
            a.sampling.top_k = r.uint();
        } else if (r.is(&.{"--top-p"})) {
            a.sampling.top_p = r.float();
        } else if (r.is(&.{"--min-p"})) {
            a.sampling.min_p = r.float();
        } else if (r.is(&.{"--repeat-penalty"})) {
            a.sampling.repeat_penalty = r.float();
        } else if (r.is(&.{"--seed"})) {
            a.sampling.seed = r.uint();
        } else if (r.is(&.{"--draft"})) {
            a.draft = r.uint();
        } else if (r.is(&.{"--draft-model"})) {
            a.draft_model = r.str();
        } else if (r.is(&.{"--server"})) {
            a.server = true;
        } else if (r.is(&.{"--port"})) {
            a.server = true;
            a.port = @intCast(r.uint());
        } else if (r.is(&.{"--no-prune"})) {
            a.load.prune_vocab = false;
        } else if (r.is(&.{"--no-repack"})) {
            a.load.repack = false;
        } else if (r.is(&.{"--cache"})) {
            const v = r.str();
            if (std.mem.eql(u8, v, "q8_0")) {
                a.cache_q8 = true;
            } else if (!std.mem.eql(u8, v, "f16")) fail("--cache expects f16 or q8_0, got '{s}'", .{v});
        } else if (r.is(&.{"--bench"})) {
            var it = std.mem.splitScalar(u8, r.str(), ',');
            const pp = std.fmt.parseInt(usize, it.next().?, 10) catch fail("--bench expects PP,TG", .{});
            const tg = std.fmt.parseInt(usize, it.next() orelse "", 10) catch fail("--bench expects PP,TG", .{});
            a.bench = .{ pp, tg };
        } else if (r.is(&.{"--chat"})) {
            a.chat = true;
        } else if (r.is(&.{"--no-think"})) {
            a.no_think = true;
        } else if (r.is(&.{ "-v", "--verbose" })) {
            a.verbose = true;
        } else if (r.is(&.{ "-h", "--help" })) {
            std.debug.print("{s}", .{usage});
            std.process.exit(0);
        } else fail("unknown option {s}\n\n{s}", .{ argv[r.i], usage });
    }
    if (a.model == null) fail("{s}", .{usage});
    return a;
}

/// Command line, then the model's general.sampling.* metadata, then defaults.
fn samplingParams(cli: SamplingArgs, f: *const gguf.File, default_seed: u64) sampler_mod.Params {
    const meta = struct {
        fn float(file: *const gguf.File, k: []const u8) ?f32 {
            return if (file.getFloat(k)) |v| @floatCast(v) else null;
        }
        fn uint(file: *const gguf.File, k: []const u8) ?usize {
            return if (file.getUint(k)) |v| @intCast(v) else null;
        }
    };
    const d: sampler_mod.Params = .{};
    return .{
        .temp = cli.temp orelse meta.float(f, "general.sampling.temp") orelse d.temp,
        .top_k = cli.top_k orelse meta.uint(f, "general.sampling.top_k") orelse d.top_k,
        .top_p = cli.top_p orelse meta.float(f, "general.sampling.top_p") orelse d.top_p,
        .min_p = cli.min_p orelse meta.float(f, "general.sampling.min_p") orelse d.min_p,
        .repeat_penalty = cli.repeat_penalty orelse meta.float(f, "general.sampling.penalty_repeat") orelse d.repeat_penalty,
        .seed = cli.seed orelse default_seed,
    };
}

/// Best-effort stdout sink for Gen.run: a closed stdout must not abort generation.
const Out = struct {
    w: *std.Io.Writer,
    pub fn emit(o: Out, bytes: []const u8) !void {
        o.w.writeAll(bytes) catch {};
        o.w.flush() catch {};
    }
};

const Template = enum {
    llama3,
    chatml,
    /// ChatML preceded by <|startoftext|> (Liquid LFM).
    lfm,
    plain,
};

fn detectTemplate(tok: *const Tokenizer) Template {
    if (tok.tokenId("<|start_header_id|>") != null and tok.tokenId("<|eot_id|>") != null) return .llama3;
    if (tok.tokenId("<|im_start|>") != null and tok.tokenId("<|im_end|>") != null)
        return if (tok.tokenId("<|startoftext|>") != null) .lfm else .chatml;
    return .plain;
}

fn appendTurn(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), tpl: Template, role: []const u8, msg: []const u8) !void {
    switch (tpl) {
        .llama3 => try buf.print(gpa, "<|start_header_id|>{s}<|end_header_id|>\n\n{s}<|eot_id|>", .{ role, msg }),
        .chatml, .lfm => try buf.print(gpa, "<|im_start|>{s}\n{s}<|im_end|>\n", .{ role, msg }),
        .plain => try buf.print(gpa, "{s}: {s}\n", .{ role, msg }),
    }
}

fn appendAssistantHeader(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), tpl: Template, has_think: bool, no_think: bool) !void {
    try buf.appendSlice(gpa, switch (tpl) {
        .llama3 => "<|start_header_id|>assistant<|end_header_id|>\n\n",
        .chatml, .lfm => "<|im_start|>assistant\n",
        .plain => "assistant:",
    });
    if (!has_think) return;
    // Mirrors each template's enable_thinking switch.
    switch (tpl) {
        .chatml => if (no_think) try buf.appendSlice(gpa, "<think>\n\n</think>\n\n"),
        .lfm => try buf.appendSlice(gpa, if (no_think) "<think></think>" else "<think>"),
        else => {},
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.smp_allocator;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    var args = parseArgs(argv);

    const nth = if (args.threads > 0) args.threads else pool_mod.defaultThreads(io);
    const pool = try pool_mod.Pool.create(gpa, io, nth);
    defer pool.destroy(gpa);
    var timer: Timer = .start(io);
    const model = model_mod.Model.load(gpa, io, args.model.?, pool, args.load) catch |e| fail("failed to load model: {s}", .{@errorName(e)});
    defer model.destroy(gpa);
    var tok = Tokenizer.init(gpa, &model.file) catch |e| fail("failed to load tokenizer: {s}", .{@errorName(e)});
    defer tok.deinit(gpa);
    const c = &model.cfg;
    const bench_need = if (args.bench) |b| b[0] + b[1] else 0;
    const default_ctx = @max(bench_need, 4096);
    const ctx = if (args.ctx > 0) args.ctx else @min(default_ctx, c.ctx_train);

    var st = model_mod.State.init(gpa, model, pool, ctx, @min(args.batch, ctx), args.cache_q8) catch |e| fail("failed to init state: {s}", .{@errorName(e)});
    defer st.deinit();
    const time_seed: u64 = @truncate(@as(u96, @bitCast(std.Io.Clock.real.now(io).nanoseconds)));
    const sp = samplingParams(args.sampling, &model.file, time_seed);
    var smp = try sampler_mod.Sampler.init(gpa, c.vocab, sp);
    defer smp.deinit(gpa);

    // Draft model for speculative decoding: same tokenizer, own state/sampler.
    // The model must outlive its State, so it lives at main scope.
    var draft_m: ?*model_mod.Model = null;
    defer if (draft_m) |m| m.destroy(gpa);
    var draft_st: ?model_mod.State = null;
    defer if (draft_st) |*d| d.deinit();
    var draft_smp: ?sampler_mod.Sampler = null;
    defer if (draft_smp) |*d| d.deinit(gpa);
    if (args.draft_model) |dpath| {
        if (args.draft == 0) args.draft = 3;
        const dm = model_mod.Model.load(gpa, io, dpath, pool, args.load) catch |e| fail("failed to load draft model: {s}", .{@errorName(e)});
        draft_m = dm;
        if (dm.cfg.vocab != c.vocab) fail("draft model vocab {d} != target vocab {d} (tokenizers must match)", .{ dm.cfg.vocab, c.vocab });
        if (dm.cfg.ctx_train < ctx) fail("draft model ctx_train {d} < target ctx {d}", .{ dm.cfg.ctx_train, ctx });
        // Draft tokens are fed one at a time; short-conv state can only be
        // rewound within one batch, so conv drafts are not supported.
        if (dm.n_conv_layers > 0) fail("draft model must not have short-conv layers", .{});
        draft_st = model_mod.State.init(gpa, dm, pool, ctx, @min(args.batch, ctx), args.cache_q8) catch |e| fail("failed to init draft state: {s}", .{@errorName(e)});
        draft_smp = try sampler_mod.Sampler.init(gpa, dm.cfg.vocab, sp);
        if (args.verbose) std.debug.print("draft: {s} (arch={s} layers={d} dim={d} vocab={d})\n", .{ std.fs.path.stem(dpath), @tagName(dm.cfg.arch), dm.cfg.n_layers, dm.cfg.dim, dm.cfg.vocab });
    }

    if (args.verbose or args.bench != null) {
        const kv_mb = @as(f64, @floatFromInt(st.k_cache.len + st.v_cache.len)) / (1 << 20);
        std.debug.print("arch={s} layers={d} dim={d} heads={d}/{d} head_dim={d} ffn={d} vocab={d} ctx={d} (KV {d:.1} MB {s}) threads={d} ({d:.0} ms load)\n", .{
            @tagName(c.arch),                            c.n_layers, c.dim, c.n_heads, c.n_kv_heads, c.head_dim, c.ffn_dim, c.vocab, ctx, kv_mb,
            if (st.kv_q8) "q8_0" else "f16", nth,
            @as(f64, @floatFromInt(timer.read())) / 1e6,
        });
        const l0 = &model.layers[0];
        std.debug.print("layers: {d} attention + {d} short-conv | weights: ffn_up={s} ffn_down={s} output={s} layout={s}\n", .{
            model.n_attn_layers,                                                                    model.n_conv_layers, @tagName(l0.w_up.typ), @tagName(l0.w_down.typ), @tagName(model.output.typ),
            if (l0.w_up.x8) (if (model.output_coarse != null) "x8 (pruned)" else "x8") else "rows",
        });
        std.debug.print("sampling: temp={d} top_k={d} top_p={d} min_p={d} repeat_penalty={d}\n", .{ sp.temp, sp.top_k, sp.top_p, sp.min_p, sp.repeat_penalty });
    }

    var out_buf: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), io, &out_buf);
    const out: Out = .{ .w = &stdout.interface };
    var gen: gen_mod.Gen = .{
        .gpa = gpa,
        .io = io,
        .st = &st,
        .tok = &tok,
        .smp = &smp,
        .draft_k = args.draft,
        .draft_st = if (draft_st) |*d| d else null,
        .draft_smp = if (draft_smp) |*d| d else null,
    };
    defer gen.deinit();

    if (args.bench) |b| return bench(gpa, io, &st, b[0], b[1]);
    if (args.server) {
        const model_name = std.fs.path.stem(args.model.?);
        return server_mod.run(gpa, io, &gen, .{
            .port = args.port,
            .model_id = model_name,
            .no_think = args.no_think,
        });
    }

    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);

    if (args.chat) {
        const tpl = detectTemplate(&tok);
        const has_think = tok.tokenId("<think>") != null;
        var in_buf: [64 * 1024]u8 = undefined;
        var stdin = std.Io.File.stdin().reader(io, &in_buf);
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        var first = true;
        var pending: ?u32 = null;
        std.debug.print("chat template: {s}. Empty line or Ctrl-D quits.\n", .{@tagName(tpl)});
        while (true) {
            std.debug.print("\n> ", .{});
            const line = (stdin.interface.takeDelimiter('\n') catch null) orelse break;
            const msg = std.mem.trim(u8, line, " \r\t");
            if (msg.len == 0) break;
            text.clearRetainingCapacity();
            tokens.clearRetainingCapacity();
            if (pending) |p| {
                try tokens.append(gpa, p);
                if (tpl == .chatml or tpl == .lfm) try text.append(gpa, '\n');
            }
            if (first and tpl == .lfm) try text.appendSlice(gpa, "<|startoftext|>");
            if (first) if (args.system) |sys| try appendTurn(gpa, &text, tpl, "system", sys);
            try appendTurn(gpa, &text, tpl, "user", msg);
            try appendAssistantHeader(gpa, &text, tpl, has_think, args.no_think);
            try tok.encode(gpa, text.items, first, true, &tokens);
            first = false;
            pending = gen.run(tokens.items, args.n_predict, out) catch |e| switch (e) {
                error.ContextFull => {
                    std.debug.print("\n[context full]\n", .{});
                    break;
                },
                else => return e,
            };
            try out.emit("\n");
        }
        std.debug.print("\n\n", .{});
        gen.report();
        return;
    }

    const prompt = if (args.prompt_file) |p|
        try std.Io.Dir.cwd().readFileAlloc(io, p, gpa, .limited(64 << 20))
    else
        args.prompt orelse fail("no prompt: use -p, -f or --chat", .{});
    defer if (args.prompt_file != null) gpa.free(prompt);
    try tok.encode(gpa, prompt, true, true, &tokens);
    if (args.verbose) std.debug.print("prompt tokens: {any}\n", .{tokens.items});
    if (tokens.items.len == 0) fail("empty prompt", .{});
    try out.emit(prompt);
    _ = try gen.run(tokens.items, args.n_predict, out);
    std.debug.print("\n\n", .{});
    gen.report();
}

fn bench(gpa: std.mem.Allocator, io: std.Io, st: *model_mod.State, pp: usize, tg: usize) !void {
    if (pp + tg > st.ctx) fail("bench needs ctx >= {d}", .{pp + tg});
    const toks = try gpa.alloc(u32, @max(pp, 1));
    defer gpa.free(toks);
    for (toks, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % st.model.cfg.vocab);
    // Warm-up pass touches every weight page.
    _ = try st.forward(toks[0..1]);
    st.reset();

    var timer: Timer = .start(io);
    if (pp > 0) {
        _ = try st.forward(toks[0..pp]);
        const ns = timer.lap();
        std.debug.print("pp{d}: {d:.2} tok/s ({d:.1} ms)\n", .{ pp, @as(f64, @floatFromInt(pp)) * 1e9 / @as(f64, @floatFromInt(ns)), @as(f64, @floatFromInt(ns)) / 1e6 });
    }
    if (tg > 0) {
        const depth = st.pos;
        _ = timer.lap();
        var t: u32 = 1;
        for (0..tg) |_| {
            const l = try st.forward(&.{t});
            t = sampler_mod.argmax(l);
        }
        const ns = timer.lap();
        const depth_msg = if (depth > 0) std.fmt.allocPrint(gpa, " @ ctx {d}", .{depth}) catch "" else "";
        defer if (depth > 0 and depth_msg.len > 0) gpa.free(depth_msg);
        std.debug.print("tg{d}{s}: {d:.2} tok/s ({d:.2} ms/tok)\n", .{
            tg, depth_msg, @as(f64, @floatFromInt(tg)) * 1e9 / @as(f64, @floatFromInt(ns)), @as(f64, @floatFromInt(ns)) / 1e6 / @as(f64, @floatFromInt(tg)),
        });
    }
}
