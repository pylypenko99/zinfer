const std = @import("std");
const gguf = @import("gguf.zig");
const model_mod = @import("model.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const sampler_mod = @import("sampler.zig");
const pool_mod = @import("pool.zig");

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
    \\
    \\  --no-repack           keep GGUF row layout (per-row kernels; for A/B comparison)
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
    draft: usize = 0,
    threads: usize = 0,
    ctx: usize = 0,
    batch: usize = 64,
    sampling: SamplingArgs = .{},
    load: model_mod.LoadOptions = .{},
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
        } else if (r.is(&.{"--no-repack"})) {
            a.load.repack = false;
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

const Out = struct {
    w: *std.Io.Writer,
    fn emit(o: Out, bytes: []const u8) void {
        o.w.writeAll(bytes) catch {};
        o.w.flush() catch {};
    }
};

/// Monotonic stopwatch on the Io awake clock.
const Timer = struct {
    io: std.Io,
    last: std.Io.Timestamp,

    fn start(io: std.Io) Timer {
        return .{ .io = io, .last = std.Io.Clock.awake.now(io) };
    }

    fn read(t: *const Timer) u64 {
        return @intCast(t.last.durationTo(std.Io.Clock.awake.now(t.io)).nanoseconds);
    }

    fn lap(t: *Timer) u64 {
        const now = std.Io.Clock.awake.now(t.io);
        const ns: u64 = @intCast(t.last.durationTo(now).nanoseconds);
        t.last = now;
        return ns;
    }
};

const Gen = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    st: *model_mod.State,
    tok: *const Tokenizer,
    smp: *sampler_mod.Sampler,
    out: Out,
    draft_k: usize = 0,
    n_prompt: usize = 0,
    n_gen: usize = 0,
    n_drafted: usize = 0,
    n_accepted: usize = 0,
    t_prompt_ns: u64 = 0,
    t_gen_ns: u64 = 0,

    /// Feed `input`, then sample up to `n` tokens. Returns the end-of-generation
    /// token if one was sampled (it is not yet in the KV cache).
    fn run(g: *Gen, input: []const u32, n: usize) !?u32 {
        var timer: Timer = .start(g.io);
        const logits = try g.st.forward(input);
        g.t_prompt_ns += timer.lap();
        g.n_prompt += input.len;
        for (input) |t| g.smp.accept(t);

        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(g.gpa);
        try history.appendSlice(g.gpa, input);

        var cur_tok = g.smp.sample(logits);
        if (g.tok.isEog(cur_tok)) {
            g.t_gen_ns += timer.lap();
            return cur_tok;
        }
        g.out.emit(g.tok.piece(cur_tok));
        g.smp.accept(cur_tok);
        try history.append(g.gpa, cur_tok);
        g.n_gen += 1;

        var draft_buf: [model_mod.max_spec]u32 = undefined;
        var cand_buf: [model_mod.max_spec + 1]u32 = undefined;

        while (g.n_gen < n and g.st.pos < g.st.ctx) {
            const k = if (g.draft_k > 0 and g.st.pos + g.draft_k + 2 <= g.st.ctx)
                sampler_mod.PromptLookup.find(history.items, @min(g.draft_k, model_mod.max_spec), &draft_buf)
            else
                0;

            if (k == 0) {
                const l = try g.st.forward(&.{cur_tok});
                const next_tok = g.smp.sample(l);
                if (g.tok.isEog(next_tok)) {
                    g.t_gen_ns += timer.lap();
                    return next_tok;
                }
                g.out.emit(g.tok.piece(next_tok));
                g.smp.accept(next_tok);
                try history.append(g.gpa, next_tok);
                cur_tok = next_tok;
                g.n_gen += 1;
            } else {
                g.n_drafted += k;
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
                    if (pred == draft_buf[accepted_count] and !g.tok.isEog(pred) and g.n_gen < n) {
                        g.out.emit(g.tok.piece(pred));
                        g.smp.accept(pred);
                        try history.append(g.gpa, pred);
                        g.n_gen += 1;
                        g.n_accepted += 1;
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

                const next_tok = next_cur.?;
                if (g.tok.isEog(next_tok)) {
                    g.t_gen_ns += timer.lap();
                    return next_tok;
                }
                g.out.emit(g.tok.piece(next_tok));
                g.smp.accept(next_tok);
                try history.append(g.gpa, next_tok);
                cur_tok = next_tok;
                g.n_gen += 1;
            }
        }
        g.t_gen_ns += timer.lap();
        return null;
    }

    fn report(g: *const Gen) void {
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
        std.debug.print("\n\nprompt: {d} tok in {d:.1} ms ({d:.2} tok/s) | gen: {d} tok in {d:.1} ms ({d:.2} tok/s)", .{
            g.n_prompt, ms(g.t_prompt_ns), rate(g.n_prompt, g.t_prompt_ns),
            g.n_gen,    ms(g.t_gen_ns),    rate(g.n_gen, g.t_gen_ns),
        });
        if (g.n_drafted > 0) {
            std.debug.print(" | spec: {d}/{d} accepted ({d:.1}%)\n", .{
                g.n_accepted,
                g.n_drafted,
                @as(f64, @floatFromInt(g.n_accepted)) * 100.0 / @as(f64, @floatFromInt(g.n_drafted)),
            });
        } else {
            std.debug.print("\n", .{});
        }
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
    const args = parseArgs(argv);

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

    var st = try model_mod.State.init(gpa, model, pool, ctx, @min(args.batch, ctx));
    defer st.deinit();
    const time_seed: u64 = @truncate(@as(u96, @bitCast(std.Io.Clock.real.now(io).nanoseconds)));
    const sp = samplingParams(args.sampling, &model.file, time_seed);
    var smp = try sampler_mod.Sampler.init(gpa, c.vocab, sp);
    defer smp.deinit(gpa);

    if (args.verbose or args.bench != null) {
        const kv_mb = @as(f64, @floatFromInt(st.k_cache.len * 2 * 2)) / (1 << 20);
        std.debug.print("arch={s} layers={d} dim={d} heads={d}/{d} head_dim={d} ffn={d} vocab={d} ctx={d} (KV {d:.1} MB) threads={d} ({d:.0} ms load)\n", .{
            @tagName(c.arch),                            c.n_layers, c.dim, c.n_heads, c.n_kv_heads, c.head_dim, c.ffn_dim, c.vocab, ctx, kv_mb, nth,
            @as(f64, @floatFromInt(timer.read())) / 1e6,
        });
        const l0 = &model.layers[0];
        std.debug.print("layers: {d} attention + {d} short-conv | weights: ffn_up={s} ffn_down={s} output={s} layout={s}\n", .{
            model.n_attn_layers,              model.n_conv_layers, @tagName(l0.w_up.typ), @tagName(l0.w_down.typ), @tagName(model.output.typ),
            if (l0.w_up.x8) "x8" else "rows",
        });
        std.debug.print("sampling: temp={d} top_k={d} top_p={d} min_p={d} repeat_penalty={d}\n", .{ sp.temp, sp.top_k, sp.top_p, sp.min_p, sp.repeat_penalty });
    }

    var out_buf: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), io, &out_buf);
    var gen: Gen = .{ .gpa = gpa, .io = io, .st = &st, .tok = &tok, .smp = &smp, .out = .{ .w = &stdout.interface }, .draft_k = args.draft };

    if (args.bench) |b| return bench(gpa, io, &st, b[0], b[1]);

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
            pending = gen.run(tokens.items, args.n_predict) catch |e| switch (e) {
                error.ContextFull => {
                    std.debug.print("\n[context full]\n", .{});
                    break;
                },
                else => return e,
            };
            gen.out.emit("\n");
        }
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
    gen.out.emit(prompt);
    _ = try gen.run(tokens.items, args.n_predict);
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
