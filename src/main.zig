const std = @import("std");
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
    \\  -t, --threads N       worker threads (default: physical cores)
    \\  -c, --ctx N           context length (default: min(4096, trained))
    \\  -b, --batch N         prompt batch size (default 64)
    \\  --temp F              temperature, 0 = greedy (default 0.8)
    \\  --top-k N             (default 40, 0 = off)
    \\  --top-p F             (default 0.95)
    \\  --min-p F             (default 0.05)
    \\  --seed N              RNG seed (default: time)
    \\  --no-repack           keep GGUF row layout (per-row kernels; for A/B comparison)
    \\  --bench PP,TG         benchmark prompt processing of PP tokens and generation of TG tokens
    \\  -v, --verbose         print model info
    \\
;

const Args = struct {
    model: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    prompt_file: ?[]const u8 = null,
    system: ?[]const u8 = null,
    n_predict: usize = 256,
    chat: bool = false,
    threads: usize = 0,
    ctx: usize = 0,
    batch: usize = 64,
    sp: sampler_mod.Params = .{},
    seed_set: bool = false,
    load: model_mod.LoadOptions = .{},
    bench: ?[2]usize = null,
    verbose: bool = false,
};

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print(fmt ++ "\n", args);
    std.process.exit(1);
}

fn parseArgs(argv: []const [:0]const u8) Args {
    var a: Args = .{};
    var i: usize = 1;
    const next = struct {
        fn f(v: []const [:0]const u8, j: *usize) []const u8 {
            j.* += 1;
            if (j.* >= v.len) fail("missing value for {s}", .{v[j.* - 1]});
            return v[j.*];
        }
    }.f;
    const num = struct {
        fn u(s: []const u8) usize {
            return std.fmt.parseInt(usize, s, 10) catch fail("bad number '{s}'", .{s});
        }
        fn f(s: []const u8) f32 {
            return std.fmt.parseFloat(f32, s) catch fail("bad number '{s}'", .{s});
        }
    };
    while (i < argv.len) : (i += 1) {
        const o = argv[i];
        const eq = std.mem.eql;
        if (eq(u8, o, "-m") or eq(u8, o, "--model")) a.model = next(argv, &i) else if (eq(u8, o, "-p") or eq(u8, o, "--prompt")) a.prompt = next(argv, &i) else if (eq(u8, o, "-f") or eq(u8, o, "--file")) a.prompt_file = next(argv, &i) else if (eq(u8, o, "-s") or eq(u8, o, "--system")) a.system = next(argv, &i) else if (eq(u8, o, "-n") or eq(u8, o, "--predict")) a.n_predict = num.u(next(argv, &i)) else if (eq(u8, o, "-t") or eq(u8, o, "--threads")) a.threads = num.u(next(argv, &i)) else if (eq(u8, o, "-c") or eq(u8, o, "--ctx")) a.ctx = num.u(next(argv, &i)) else if (eq(u8, o, "-b") or eq(u8, o, "--batch")) a.batch = @max(1, num.u(next(argv, &i))) else if (eq(u8, o, "--temp")) a.sp.temp = num.f(next(argv, &i)) else if (eq(u8, o, "--top-k")) a.sp.top_k = num.u(next(argv, &i)) else if (eq(u8, o, "--top-p")) a.sp.top_p = num.f(next(argv, &i)) else if (eq(u8, o, "--min-p")) a.sp.min_p = num.f(next(argv, &i)) else if (eq(u8, o, "--seed")) {
            a.sp.seed = num.u(next(argv, &i));
            a.seed_set = true;
        } else if (eq(u8, o, "--no-repack")) {
            a.load.repack = false;
        } else if (eq(u8, o, "--bench")) {
            const v = next(argv, &i);
            var it = std.mem.splitScalar(u8, v, ',');
            const pp = num.u(it.next().?);
            const tg = num.u(it.next() orelse fail("--bench expects PP,TG", .{}));
            a.bench = .{ pp, tg };
        } else if (eq(u8, o, "--chat")) a.chat = true else if (eq(u8, o, "-v") or eq(u8, o, "--verbose")) a.verbose = true else if (eq(u8, o, "-h") or eq(u8, o, "--help")) {
            std.debug.print("{s}", .{usage});
            std.process.exit(0);
        } else fail("unknown option {s}\n\n{s}", .{ o, usage });
    }
    if (a.model == null) fail("{s}", .{usage});
    return a;
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
    io: std.Io,
    st: *model_mod.State,
    tok: *const Tokenizer,
    smp: *sampler_mod.Sampler,
    out: Out,
    n_prompt: usize = 0,
    n_gen: usize = 0,
    t_prompt_ns: u64 = 0,
    t_gen_ns: u64 = 0,

    /// Feed `input`, then sample up to `n` tokens. Returns the end-of-generation
    /// token if one was sampled (it is not yet in the KV cache).
    fn run(g: *Gen, input: []const u32, n: usize) !?u32 {
        var timer: Timer = .start(g.io);
        var logits = try g.st.forward(input);
        g.t_prompt_ns += timer.lap();
        g.n_prompt += input.len;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const tok = g.smp.sample(logits);
            if (g.tok.isEog(tok)) {
                g.t_gen_ns += timer.lap();
                return tok;
            }
            g.out.emit(g.tok.piece(tok));
            if (g.st.pos >= g.st.ctx) {
                g.out.emit("\n[context full]\n");
                break;
            }
            logits = try g.st.forward(&.{tok});
            g.n_gen += 1;
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
        std.debug.print("\n\nprompt: {d} tok in {d:.1} ms ({d:.2} tok/s) | gen: {d} tok in {d:.1} ms ({d:.2} tok/s)\n", .{
            g.n_prompt, ms(g.t_prompt_ns), rate(g.n_prompt, g.t_prompt_ns),
            g.n_gen,    ms(g.t_gen_ns),    rate(g.n_gen, g.t_gen_ns),
        });
    }
};

const Template = enum { llama3, chatml, plain };

fn detectTemplate(tok: *const Tokenizer) Template {
    if (tok.tokenId("<|start_header_id|>") != null and tok.tokenId("<|eot_id|>") != null) return .llama3;
    if (tok.tokenId("<|im_start|>") != null and tok.tokenId("<|im_end|>") != null) return .chatml;
    return .plain;
}

fn appendTurn(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), tpl: Template, role: []const u8, msg: []const u8) !void {
    switch (tpl) {
        .llama3 => try buf.print(gpa, "<|start_header_id|>{s}<|end_header_id|>\n\n{s}<|eot_id|>", .{ role, msg }),
        .chatml => try buf.print(gpa, "<|im_start|>{s}\n{s}<|im_end|>\n", .{ role, msg }),
        .plain => try buf.print(gpa, "{s}: {s}\n", .{ role, msg }),
    }
}

fn appendAssistantHeader(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), tpl: Template) !void {
    try buf.appendSlice(gpa, switch (tpl) {
        .llama3 => "<|start_header_id|>assistant<|end_header_id|>\n\n",
        .chatml => "<|im_start|>assistant\n",
        .plain => "assistant:",
    });
}

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.smp_allocator;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    var args = parseArgs(argv);
    if (!args.seed_set) args.sp.seed = @truncate(@as(u96, @bitCast(std.Io.Clock.real.now(io).nanoseconds)));

    const nth = if (args.threads > 0) args.threads else pool_mod.defaultThreads(io);
    const pool = try pool_mod.Pool.create(gpa, io, nth);
    defer pool.destroy(gpa);
    var timer: Timer = .start(io);
    const model = model_mod.Model.load(gpa, io, args.model.?, pool, args.load) catch |e| fail("failed to load model: {s}", .{@errorName(e)});
    defer model.destroy(gpa);
    var tok = Tokenizer.init(gpa, &model.file) catch |e| fail("failed to load tokenizer: {s}", .{@errorName(e)});
    defer tok.deinit(gpa);
    const c = &model.cfg;
    const ctx = if (args.ctx > 0) args.ctx else @min(4096, c.ctx_train);

    var st = try model_mod.State.init(gpa, model, pool, ctx, @min(args.batch, ctx));
    defer st.deinit();
    var smp = try sampler_mod.Sampler.init(gpa, c.vocab, args.sp);
    defer smp.deinit(gpa);

    if (args.verbose or args.bench != null) {
        std.debug.print("arch={s} layers={d} dim={d} heads={d}/{d} head_dim={d} ffn={d} vocab={d} ctx={d} threads={d} ({d:.0} ms load)\n", .{
            @tagName(c.arch),                            c.n_layers, c.dim, c.n_heads, c.n_kv_heads, c.head_dim, c.ffn_dim, c.vocab, ctx, nth,
            @as(f64, @floatFromInt(timer.read())) / 1e6,
        });
        const l0 = &model.layers[0];
        std.debug.print("weights: wq={s} ffn_down={s} output={s} layout={s}\n", .{ @tagName(l0.wq.typ), @tagName(l0.w_down.typ), @tagName(model.output.typ), if (l0.wq.x8) "x8" else "rows" });
    }

    var out_buf: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), io, &out_buf);
    var gen: Gen = .{ .io = io, .st = &st, .tok = &tok, .smp = &smp, .out = .{ .w = &stdout.interface } };

    if (args.bench) |b| return bench(gpa, io, &st, b[0], b[1]);

    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);

    if (args.chat) {
        const tpl = detectTemplate(&tok);
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
                if (tpl == .chatml) try text.append(gpa, '\n');
            }
            if (first) if (args.system) |sys| try appendTurn(gpa, &text, tpl, "system", sys);
            try appendTurn(gpa, &text, tpl, "user", msg);
            try appendAssistantHeader(gpa, &text, tpl);
            try tok.encode(gpa, text.items, first, true, &tokens);
            first = false;
            pending = gen.run(tokens.items, args.n_predict) catch |e| switch (e) {
                error.ContextFull => {
                    std.debug.print("\n[context full]\n", .{});
                    break;
                },
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
    st.pos = 0;

    var timer: Timer = .start(io);
    if (pp > 0) {
        _ = try st.forward(toks[0..pp]);
        const ns = timer.lap();
        std.debug.print("pp{d}: {d:.2} tok/s ({d:.1} ms)\n", .{ pp, @as(f64, @floatFromInt(pp)) * 1e9 / @as(f64, @floatFromInt(ns)), @as(f64, @floatFromInt(ns)) / 1e6 });
    }
    if (tg > 0) {
        st.pos = 0;
        _ = timer.lap();
        var t: u32 = 1;
        for (0..tg) |_| {
            const l = try st.forward(&.{t});
            t = sampler_mod.argmax(l);
        }
        const ns = timer.lap();
        std.debug.print("tg{d}: {d:.2} tok/s ({d:.2} ms/tok)\n", .{ tg, @as(f64, @floatFromInt(tg)) * 1e9 / @as(f64, @floatFromInt(ns)), @as(f64, @floatFromInt(ns)) / 1e6 / @as(f64, @floatFromInt(tg)) });
    }
}
