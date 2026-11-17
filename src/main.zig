const std = @import("std");
const builtin = @import("builtin");
const gguf = @import("gguf.zig");
const model_mod = @import("model.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const sampler_mod = @import("sampler.zig");
const pool_mod = @import("pool.zig");
const server_mod = @import("server.zig");
const gen_mod = @import("gen.zig");
const chat_mod = @import("chat.zig");
const platform = @import("platform.zig");
const Timer = gen_mod.Timer;

const usage =
    \\usage: zinfer -m MODEL.gguf [options]
    \\
    \\  -p, --prompt TEXT     prompt text (special tokens like <|eot_id|> are parsed)
    \\  -f, --file PATH       read prompt from file
    \\  -n, --predict N       max tokens to generate (default: 256 for prompt, unlimited for --chat; -1 or 0 = unlimited)
    \\  --chat                interactive chat using the model's chat template
    \\  -s, --system TEXT     system message for --chat
    \\  --no-think            start assistant turns with an empty think block (Qwen3, LFM3)
    \\  -t, --threads N       worker threads (default: physical cores)
    \\  -c, --ctx N           context length (default: min(4096, trained))
    \\  -b, --batch N         prompt batch size (default 512)
    \\
    \\sampling (defaults: the model's general.sampling.* metadata, else the values shown)
    \\  --temp F              temperature, 0 = greedy (0.8)
    \\  --top-k N             0 = off (40)
    \\  --top-p F             (0.95)
    \\  --min-p F             (0.05)
    \\  --repeat-penalty F    penalize the last 64 tokens, 1 = off (1.0)
    \\  --dry-multiplier F    DRY repetition penalty multiplier, 0 = off (default: 0.0, 0.8 recommended)
    \\  --dry-base F          DRY exponential penalty base (default: 1.75)
    \\  --dry-allowed-length N minimum matching sequence length before DRY penalty (default: 2)
    \\  --dry-range N         DRY lookback context length in tokens (default: 1024)
    \\  --xtc-probability F   XTC chance to exclude top choices, 0 = off (default: 0.0, 0.5 recommended)
    \\  --xtc-threshold F     XTC minimum token probability threshold to be considered (default: 0.1, <= 0.5)
    \\  --xtc-min-keep N      XTC minimum tokens to keep after exclusion (default: 1)
    \\  --seed N              RNG seed (time)
    \\
    \\prompt caching:
    \\  --prompt-cache PATH   load and save prompt cache snapshot (KV cache)
    \\  --prompt-cache-all    save generated tokens to cache as well
    \\  --no-pin              disable CPU thread pinning / core affinity
    \\
    \\speculative decoding:
    \\  --draft, --lookup N   n-gram prompt lookup draft tokens, 0 = off (default 0, 3-5 recommended)
    \\  --draft-model PATH    small model as draft; --draft sets lookahead (default 3)
    \\                        (draft must share the target's tokenizer)
    \\  --no-adaptive-draft   disable real-time dynamic draft length adaptation
    \\
    \\  --server              run OpenAI / llama.cpp-compatible HTTP server
    \\  --port N              server port (default 8080; implies --server)
    \\  --json                enforce valid JSON object output (grammar-constrained decoding)
    \\  --no-prune            keep full lm_head matmul (for A/B comparison)
    \\  --no-repack           keep GGUF row layout (per-row kernels; for A/B comparison)
    \\  --cache f16|q8_0|q4_0|q2_0|q2_k K/V cache format; q4_0 (3.6x),
    \\                        q2_k (4.6x hybrid 2b K / 4b V), q2_0 (6.4x)
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
    dry_multiplier: ?f32 = null,
    dry_base: ?f32 = null,
    dry_allowed_length: ?usize = null,
    dry_range: ?usize = null,
    xtc_probability: ?f32 = null,
    xtc_threshold: ?f32 = null,
    xtc_min_keep: ?usize = null,
    seed: ?u64 = null,
};

const Args = struct {
    model: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    prompt_file: ?[]const u8 = null,
    prompt_cache: ?[]const u8 = null,
    prompt_cache_all: bool = false,
    system: ?[]const u8 = null,
    n_predict: ?usize = null,
    chat: bool = false,
    no_think: bool = false,
    json: bool = false,
    server: bool = false,
    port: u16 = 8080,
    draft: usize = 0,
    draft_model: ?[]const u8 = null,
    draft_adaptive: bool = true,
    threads: usize = 0,
    pin: bool = true,
    ctx: usize = 0,
    batch: usize = 512,
    sampling: SamplingArgs = .{},
    load: model_mod.LoadOptions = .{},
    cache_type: model_mod.CacheType = .f16,
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
        } else if (r.is(&.{"--prompt-cache"})) {
            a.prompt_cache = r.str();
        } else if (r.is(&.{"--prompt-cache-all"})) {
            a.prompt_cache_all = true;
        } else if (r.is(&.{ "-s", "--system" })) {
            a.system = r.str();
        } else if (r.is(&.{ "-n", "--predict" })) {
            const s = r.str();
            if (std.mem.eql(u8, s, "-1") or std.mem.eql(u8, s, "0") or std.mem.eql(u8, s, "inf")) {
                a.n_predict = 0;
            } else {
                a.n_predict = std.fmt.parseInt(usize, s, 10) catch fail("bad number '{s}'", .{s});
            }
        } else if (r.is(&.{ "-t", "--threads" })) {
            a.threads = r.uint();
        } else if (r.is(&.{"--no-pin"})) {
            a.pin = false;
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
        } else if (r.is(&.{"--dry-multiplier"})) {
            a.sampling.dry_multiplier = r.float();
        } else if (r.is(&.{"--dry-base"})) {
            a.sampling.dry_base = r.float();
        } else if (r.is(&.{"--dry-allowed-length"})) {
            a.sampling.dry_allowed_length = r.uint();
        } else if (r.is(&.{"--dry-range"})) {
            a.sampling.dry_range = r.uint();
        } else if (r.is(&.{ "--xtc-probability", "--xtc-p" })) {
            a.sampling.xtc_probability = r.float();
        } else if (r.is(&.{ "--xtc-threshold", "--xtc-t" })) {
            a.sampling.xtc_threshold = r.float();
        } else if (r.is(&.{"--xtc-min-keep"})) {
            a.sampling.xtc_min_keep = r.uint();
        } else if (r.is(&.{"--seed"})) {
            a.sampling.seed = r.uint();
        } else if (r.is(&.{ "--draft", "--lookup" })) {
            a.draft = r.uint();
        } else if (r.is(&.{"--draft-model"})) {
            a.draft_model = r.str();
        } else if (r.is(&.{"--no-adaptive-draft"})) {
            a.draft_adaptive = false;
        } else if (r.is(&.{"--draft-adaptive"})) {
            a.draft_adaptive = true;
        } else if (r.is(&.{"--server"})) {
            a.server = true;
        } else if (r.is(&.{"--json"})) {
            a.json = true;
        } else if (r.is(&.{"--port"})) {
            a.server = true;
            a.port = std.math.cast(u16, r.uint()) orelse fail("port out of range (1-65535)", .{});
        } else if (r.is(&.{"--no-prune"})) {
            a.load.prune_vocab = false;
        } else if (r.is(&.{"--no-repack"})) {
            a.load.repack = false;
        } else if (r.is(&.{"--cache"})) {
            const v = r.str();
            if (std.mem.eql(u8, v, "q8_0") or std.mem.eql(u8, v, "q8")) {
                a.cache_type = .q8_0;
            } else if (std.mem.eql(u8, v, "q4_0") or std.mem.eql(u8, v, "q4")) {
                a.cache_type = .q4_0;
            } else if (std.mem.eql(u8, v, "q2_0") or std.mem.eql(u8, v, "q2")) {
                a.cache_type = .q2_0;
            } else if (std.mem.eql(u8, v, "q2_k") or std.mem.eql(u8, v, "q2k")) {
                a.cache_type = .q2_k;
            } else if (std.mem.eql(u8, v, "f16")) {
                a.cache_type = .f16;
            } else fail("--cache expects f16, q8_0, q4_0, q2_0, or q2_k, got '{s}'", .{v});
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
        .dry_multiplier = cli.dry_multiplier orelse d.dry_multiplier,
        .dry_base = cli.dry_base orelse d.dry_base,
        .dry_allowed_length = cli.dry_allowed_length orelse d.dry_allowed_length,
        .dry_range = cli.dry_range orelse d.dry_range,
        .xtc_probability = cli.xtc_probability orelse meta.float(f, "general.sampling.xtc_probability") orelse d.xtc_probability,
        .xtc_threshold = cli.xtc_threshold orelse meta.float(f, "general.sampling.xtc_threshold") orelse d.xtc_threshold,
        .xtc_min_keep = cli.xtc_min_keep orelse meta.uint(f, "general.sampling.xtc_min_keep") orelse d.xtc_min_keep,
        .seed = cli.seed orelse default_seed,
    };
}

/// Best-effort stdout sink for Gen.run: a closed stdout must not abort generation.
const Out = struct {
    w: *std.Io.Writer,
    verbose: bool = false,
    pub fn emit(o: Out, bytes: []const u8) !void {
        o.w.writeAll(bytes) catch |err| {
            if (o.verbose and err != error.BrokenPipe) {
                std.debug.print("\n[warning: stdout write failed: {s}]\n", .{@errorName(err)});
            }
        };
        o.w.flush() catch |err| {
            if (o.verbose and err != error.BrokenPipe) {
                std.debug.print("\n[warning: stdout flush failed: {s}]\n", .{@errorName(err)});
            }
        };
    }
};

const PosixSigType = if (@hasDecl(std.posix, "SIG")) std.posix.SIG else c_int;
fn sigHandler(_: PosixSigType) callconv(.c) void {
    gen_mod.interrupted.store(true, .monotonic);
}

fn winCtrlHandler(ctrl_type: u32) callconv(.winapi) i32 {
    if (ctrl_type == 0 or ctrl_type == 1) { // CTRL_C_EVENT or CTRL_BREAK_EVENT
        gen_mod.interrupted.store(true, .monotonic);
        return 1;
    }
    return 0;
}

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.smp_allocator;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    var args = parseArgs(argv);

    const nth = if (args.threads > 0) args.threads else pool_mod.defaultThreads(io);
    const pool = try pool_mod.Pool.create(gpa, io, nth, args.pin);
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

    var st = model_mod.State.init(gpa, model, pool, ctx, @min(args.batch, ctx), args.cache_type) catch |e| fail("failed to init state: {s}", .{@errorName(e)});
    defer st.deinit();
    const time_seed: u64 = @truncate(@as(u96, @bitCast(std.Io.Clock.real.now(io).nanoseconds)));
    const sp = samplingParams(args.sampling, &model.file, time_seed);
    var smp = try sampler_mod.Sampler.init(gpa, c.vocab, sp);
    defer smp.deinit(gpa);

    // Default sequence breaker for DRY sampler
    var nl_toks: std.ArrayList(u32) = .empty;
    defer nl_toks.deinit(gpa);
    tok.encode(gpa, "\n", false, false, &nl_toks) catch {};
    for (nl_toks.items) |bt| smp.addBreaker(bt);

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
        draft_st = model_mod.State.init(gpa, dm, pool, ctx, @min(args.batch, ctx), args.cache_type) catch |e| fail("failed to init draft state: {s}", .{@errorName(e)});
        draft_smp = try sampler_mod.Sampler.init(gpa, dm.cfg.vocab, sp);
        if (args.verbose) std.debug.print("draft: {s} (arch={s} layers={d} dim={d} vocab={d})\n", .{ std.fs.path.stem(dpath), @tagName(dm.cfg.arch), dm.cfg.n_layers, dm.cfg.dim, dm.cfg.vocab });
    }

    if (args.verbose or args.bench != null) {
        const kv_mb = @as(f64, @floatFromInt(st.k_cache.len + st.v_cache.len)) / (1 << 20);
        const cache_name = switch (st.cache_type) {
            .f16 => "f16",
            .q8_0 => "q8_0",
            .q4_0 => "q4_0",
            .q2_0 => "q2_0",
            .q2_k => "q2_k",
        };
        std.debug.print("arch={s} layers={d} dim={d} heads={d}/{d} head_dim={d} ffn={d} vocab={d} ctx={d} (KV {d:.1} MB {s}) threads={d}", .{
            @tagName(c.arch), c.n_layers, c.dim, c.n_heads, c.n_kv_heads, c.head_dim, c.ffn_dim, c.vocab, ctx, kv_mb,
            cache_name,       nth,
        });
        if (args.pin) {
            std.debug.print(" (pinned to CPUs: ", .{});
            for (pool.cpu_map, 0..) |cpu, i| {
                if (i > 0) std.debug.print(",", .{});
                std.debug.print("{d}", .{cpu});
            }
            std.debug.print(")", .{});
        }
        std.debug.print(" ({d:.0} ms load)\n", .{@as(f64, @floatFromInt(timer.read())) / 1e6});
        const l0 = &model.layers[0];
        std.debug.print("layers: {d} attention + {d} short-conv | weights: ffn_up={s} ffn_down={s} output={s} layout={s}", .{
            model.n_attn_layers,                                                                          model.n_conv_layers, @tagName(l0.ffnUp().typ), @tagName(l0.ffnDown().typ), @tagName(model.output.typ),
            if (l0.ffnUp().x8) (if (model.output_coarse != null) "x8 (pruned)" else "x8") else "rows",
        });
        if (c.n_experts > 0) {
            std.debug.print(" | moe={d}x{d} (top-{d})", .{ c.n_experts, c.expert_feed_forward_length, c.n_experts_used });
        }
        std.debug.print("\n", .{});
        std.debug.print("sampling: temp={d} top_k={d} top_p={d} min_p={d} repeat_penalty={d}", .{ sp.temp, sp.top_k, sp.top_p, sp.min_p, sp.repeat_penalty });
        if (sp.dry_multiplier > 0) {
            std.debug.print(" dry={d:.2}x{d:.2}^{d}", .{ sp.dry_multiplier, sp.dry_base, sp.dry_allowed_length });
        }
        if (sp.xtc_probability > 0) {
            std.debug.print(" xtc=p:{d:.2},t:{d:.2}", .{ sp.xtc_probability, sp.xtc_threshold });
        }
        std.debug.print("\n", .{});
    }

    var out_buf: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), io, &out_buf);
    const out: Out = .{ .w = &stdout.interface, .verbose = args.verbose };
    var gen: gen_mod.Gen = .{
        .gpa = gpa,
        .io = io,
        .st = &st,
        .tok = &tok,
        .smp = &smp,
        .draft_k = args.draft,
        .draft_adaptive = args.draft_adaptive,
        .draft_st = if (draft_st) |*d| d else null,
        .draft_smp = if (draft_smp) |*d| d else null,
    };
    defer gen.deinit();

    if (args.json) gen.json_validator = gen_mod.grammar_mod.JsonValidator.initObjectOnly();

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
        var cur_n_predict: usize = args.n_predict orelse 0;
        if (builtin.os.tag == .windows) {
            platform.setupCtrlHandler(winCtrlHandler);
        } else if (@hasDecl(std.posix, "sigaction") and @hasDecl(std.posix, "SIG")) {
            var act: std.posix.Sigaction = std.mem.zeroes(std.posix.Sigaction);
            act.handler = .{ .handler = sigHandler };
            std.posix.sigaction(std.posix.SIG.INT, &act, null);
        }
        const tpl: chat_mod.Template = .detect(&tok);
        const has_think = chat_mod.hasThinkSwitch(&tok, model.file.getString("tokenizer.chat_template"));
        var in_buf: [64 * 1024]u8 = undefined;
        var stdin = std.Io.File.stdin().reader(io, &in_buf);
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        var chat: chat_mod.Writer = .{ .gpa = gpa, .buf = &text, .tpl = tpl };
        defer chat.deinit();
        var first = true;
        var pending: ?u32 = null;

        if (args.prompt_cache) |cp| {
            const exists = blk: {
                const f = std.Io.Dir.cwd().openFile(io, cp, .{}) catch break :blk false;
                f.close(io);
                break :blk true;
            };
            if (exists) {
                gen.loadCache(cp) catch |e| fail("failed to load prompt cache '{s}': {s}", .{ cp, @errorName(e) });
                first = false;
                if (args.verbose) {
                    std.debug.print("loaded chat cache '{s}' ({d} tokens, pos {d})\n", .{ cp, gen.cached.items.len, st.pos });
                }
            }
        }

        std.debug.print(
            \\
            \\  \x1b[1;36mzinfer interactive chat\x1b[0m
            \\  template: \x1b[33m{s}\x1b[0m | cache: \x1b[35m{s}\x1b[0m | threads: {d} | limit: {s}
            \\  Type \x1b[32m/help\x1b[0m for commands, Ctrl+C to stop generation, \x1b[31m/quit\x1b[0m or Ctrl-D to exit.
            \\
        , .{ @tagName(tpl), @tagName(st.cache_type), pool.nth, if (cur_n_predict == 0) "unlimited" else "custom" });
        while (true) {
            std.debug.print("\n\x1b[1;32mUser>\x1b[0m ", .{});
            const line = (stdin.interface.takeDelimiter('\n') catch null) orelse break;
            const msg = std.mem.trim(u8, line, " \r\t");
            if (msg.len == 0) continue;

            if (std.mem.startsWith(u8, msg, "/")) {
                var it = std.mem.tokenizeScalar(u8, msg, ' ');
                const cmd = it.next() orelse "";
                if (std.mem.eql(u8, cmd, "/quit") or std.mem.eql(u8, cmd, "/exit")) {
                    break;
                } else if (std.mem.eql(u8, cmd, "/save")) {
                    const path = it.next() orelse args.prompt_cache orelse {
                        std.debug.print("usage: /save <path>\n", .{});
                        continue;
                    };
                    gen.saveCache(path) catch |e| {
                        std.debug.print("error saving cache to '{s}': {s}\n", .{ path, @errorName(e) });
                        continue;
                    };
                    std.debug.print("saved cache to '{s}' ({d} tokens, pos {d})\n", .{ path, gen.cached.items.len, st.pos });
                    continue;
                } else if (std.mem.eql(u8, cmd, "/load")) {
                    const path = it.next() orelse args.prompt_cache orelse {
                        std.debug.print("usage: /load <path>\n", .{});
                        continue;
                    };
                    gen.loadCache(path) catch |e| {
                        std.debug.print("error loading cache from '{s}': {s}\n", .{ path, @errorName(e) });
                        continue;
                    };
                    first = false;
                    pending = null;
                    std.debug.print("loaded cache from '{s}' ({d} tokens, pos {d})\n", .{ path, gen.cached.items.len, st.pos });
                    continue;
                } else if (std.mem.eql(u8, cmd, "/reset") or std.mem.eql(u8, cmd, "/clear")) {
                    gen.reset();
                    first = true;
                    pending = null;
                    std.debug.print("context cleared.\n", .{});
                    continue;
                } else if (std.mem.startsWith(u8, cmd, "/predict") or std.mem.startsWith(u8, cmd, "/tokens")) {
                    var pit = std.mem.tokenizeScalar(u8, msg, ' ');
                    _ = pit.next();
                    if (pit.next()) |val| {
                        if (std.mem.eql(u8, val, "-1") or std.mem.eql(u8, val, "0") or std.mem.eql(u8, val, "inf")) {
                            cur_n_predict = 0;
                            std.debug.print("token limit per turn: unlimited (until EOS or context full)\n", .{});
                        } else if (std.fmt.parseInt(usize, val, 10)) |p_val| {
                            cur_n_predict = p_val;
                            std.debug.print("token limit per turn set to {d}\n", .{cur_n_predict});
                        } else |_| {
                            std.debug.print("invalid number '{s}'. Usage: /predict [N|0|-1]\n", .{val});
                        }
                    } else {
                        if (cur_n_predict == 0) {
                            std.debug.print("current token limit: unlimited (until EOS or context full)\n", .{});
                        } else {
                            std.debug.print("current token limit: {d} tokens per turn\n", .{cur_n_predict});
                        }
                    }
                    continue;
                } else if (std.mem.eql(u8, cmd, "/stats")) {
                    gen.report();
                    std.debug.print("cached tokens: {d}, pos: {d}/{d}\n", .{ gen.cached.items.len, st.pos, st.ctx });
                    continue;
                } else if (std.mem.eql(u8, cmd, "/help")) {
                    std.debug.print(
                        \\Commands:
                        \\  /predict [N]   Set max tokens per turn (0 or -1 for unlimited)
                        \\  /save [path]   Save current KV cache and token history to file
                        \\  /load [path]   Load KV cache and token history from file
                        \\  /reset, /clear Reset context and start a new conversation
                        \\  /stats         Display generation timing and memory statistics
                        \\  /quit, /exit   Exit chat
                        \\  /help          Show this help message
                        \\
                    , .{});
                    continue;
                } else {
                    std.debug.print("unknown command '{s}'. Type /help for available commands.\n", .{cmd});
                    continue;
                }
            }

            text.clearRetainingCapacity();
            tokens.clearRetainingCapacity();
            if (pending) |p| {
                try tokens.append(gpa, p);
                try text.appendSlice(gpa, tpl.turnSeparator());
            } else if (!first) {
                if (tok.eog.len > 0) {
                    try tokens.append(gpa, tok.eog[0]);
                } else if (tok.eos) |eos| {
                    try tokens.append(gpa, eos);
                }
                try text.appendSlice(gpa, tpl.turnSeparator());
            }
            if (first) {
                try chat.begin();
                if (args.system) |sys| try chat.turn("system", sys);
            }
            try chat.turn("user", msg);
            try chat.assistantHeader(has_think, args.no_think);
            try tok.encode(gpa, text.items, first, true, &tokens);
            first = false;
            std.debug.print("\x1b[1;35mAssistant>\x1b[0m ", .{});
            pending = gen.run(tokens.items, cur_n_predict, out) catch |e| blk: {
                if (e != error.ContextFull) return e;
                std.debug.print("\n[context full: pos {d}/{d}. Attempting context shift...]\n", .{ st.pos, st.ctx });
                if (gen.cached.items.len > 64) {
                    // StreamingLLM: preserve initial 4 attention sink tokens to anchor softmax,
                    // plus a contiguous tail of recent conversation tokens.
                    const sink_count: usize = @min(4, gen.cached.items.len);
                    const needed = tokens.items.len + sink_count;
                    const max_tail = if (st.ctx * 3 / 4 > needed) (st.ctx * 3 / 4) - needed else st.ctx / 2;
                    const avail = gen.cached.items.len - sink_count;
                    const tail_len = @min(avail, max_tail);
                    const tail_start = gen.cached.items.len - tail_len;

                    var shifted: std.ArrayList(u32) = .empty;
                    defer shifted.deinit(gpa);
                    try shifted.appendSlice(gpa, gen.cached.items[0..sink_count]);
                    try shifted.appendSlice(gpa, gen.cached.items[tail_start..]);
                    try shifted.appendSlice(gpa, tokens.items);
                    gen.reset();
                    break :blk gen.run(shifted.items, cur_n_predict, out) catch |e2| {
                        std.debug.print("\n[StreamingLLM context shift failed: {s}. Use /reset or /clear to restart]\n", .{@errorName(e2)});
                        continue;
                    };
                } else {
                    std.debug.print("\n[context full. Use /reset or /clear to restart]\n", .{});
                    continue;
                }
            };
            try out.emit("\n");
            std.debug.print("\x1b[2;37m[{d} tokens in cache, pos {d}/{d}]\x1b[0m\n", .{ gen.cached.items.len, st.pos, st.ctx });
        }

        if (args.prompt_cache) |cp| {
            if (args.prompt_cache_all) {
                gen.saveCache(cp) catch |e| std.debug.print("failed to save cache to '{s}': {s}\n", .{ cp, @errorName(e) });
                if (args.verbose) {
                    std.debug.print("saved chat cache to '{s}' ({d} tokens, pos {d})\n", .{ cp, gen.cached.items.len, st.pos });
                }
            }
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
    if (args.prompt_cache) |cp| {
        const exists = blk: {
            const f = std.Io.Dir.cwd().openFile(io, cp, .{}) catch break :blk false;
            f.close(io);
            break :blk true;
        };
        if (exists) {
            gen.loadCache(cp) catch |e| fail("failed to load prompt cache '{s}': {s}", .{ cp, @errorName(e) });
            if (args.verbose) {
                std.debug.print("loaded prompt cache '{s}' ({d} tokens, pos {d})\n", .{ cp, gen.cached.items.len, st.pos });
            }
        } else if (!args.prompt_cache_all) {
            gen.prompt_cache_save = cp;
        }
    }

    try out.emit(prompt);
    _ = try gen.runPrompt(tokens.items, args.n_predict orelse 256, out);

    if (args.prompt_cache) |cp| {
        if (args.prompt_cache_all) {
            gen.saveCache(cp) catch |e| fail("failed to save prompt cache '{s}': {s}", .{ cp, @errorName(e) });
            if (args.verbose) {
                std.debug.print("saved prompt cache '{s}' ({d} tokens, pos {d})\n", .{ cp, gen.cached.items.len, st.pos });
            }
        }
    }
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
