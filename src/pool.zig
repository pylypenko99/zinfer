//! Persistent spin-then-sleep thread pool. `run` broadcasts one job to every
//! worker (the caller participates as worker 0) and returns when all finish.
//! Workers spin between dispatches so back-to-back ops in a forward pass cost
//! no syscalls; after a short idle period they park on a futex.
const std = @import("std");
const Atomic = std.atomic.Value;

const spin_limit = 1 << 16;

const Job = struct {
    ctx: *anyopaque,
    func: *const fn (ctx: *anyopaque, tid: usize, nth: usize) void,
};

pub const Pool = struct {
    io: std.Io,
    threads: []std.Thread,
    nth: usize,
    job: Job = undefined,
    gen: Atomic(u32) = .init(0),
    done: Atomic(u32) = .init(0),
    sleepers: Atomic(u32) = .init(0),
    quit: bool = false,

    pub fn create(gpa: std.mem.Allocator, io: std.Io, nth: usize) !*Pool {
        const self = try gpa.create(Pool);
        errdefer gpa.destroy(self);
        self.* = .{ .io = io, .threads = try gpa.alloc(std.Thread, nth - 1), .nth = nth };
        var spawned: usize = 0;
        errdefer {
            self.shutdown(spawned);
            gpa.free(self.threads);
        }
        while (spawned < nth - 1) : (spawned += 1) {
            self.threads[spawned] = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, worker, .{ self, spawned + 1 });
        }
        return self;
    }

    fn shutdown(self: *Pool, n: usize) void {
        self.quit = true;
        _ = self.gen.fetchAdd(1, .seq_cst);
        self.io.futexWake(u32, &self.gen.raw, std.math.maxInt(u32));
        for (self.threads[0..n]) |t| t.join();
    }

    pub fn destroy(self: *Pool, gpa: std.mem.Allocator) void {
        self.shutdown(self.threads.len);
        gpa.free(self.threads);
        gpa.destroy(self);
    }

    fn worker(self: *Pool, tid: usize) void {
        var seen: u32 = 0;
        while (true) {
            var spins: u32 = 0;
            while (self.gen.load(.acquire) == seen) {
                if (spins < spin_limit) {
                    std.atomic.spinLoopHint();
                    spins += 1;
                } else {
                    _ = self.sleepers.fetchAdd(1, .seq_cst);
                    if (self.gen.load(.seq_cst) == seen) self.io.futexWaitUncancelable(u32, &self.gen.raw, seen);
                    _ = self.sleepers.fetchSub(1, .seq_cst);
                }
            }
            seen = self.gen.load(.acquire);
            if (self.quit) return;
            self.job.func(self.job.ctx, tid, self.nth);
            _ = self.done.fetchAdd(1, .release);
        }
    }

    /// Run `f(ctx, tid, nth)` on all threads; `ctx` must be a mutable pointer.
    pub fn run(self: *Pool, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize, usize) void) void {
        const C = @TypeOf(ctx);
        const W = struct {
            fn call(p: *anyopaque, tid: usize, nth: usize) void {
                f(@as(C, @ptrCast(@alignCast(p))), tid, nth);
            }
        };
        if (self.nth == 1) return f(ctx, 0, 1);
        self.job = .{ .ctx = @ptrCast(ctx), .func = W.call };
        self.done.store(0, .monotonic);
        _ = self.gen.fetchAdd(1, .seq_cst);
        if (self.sleepers.load(.seq_cst) != 0) self.io.futexWake(u32, &self.gen.raw, std.math.maxInt(u32));
        f(ctx, 0, self.nth);
        const want: u32 = @intCast(self.nth - 1);
        while (self.done.load(.acquire) != want) std.atomic.spinLoopHint();
    }
};

/// Dynamic work distribution: threads claim `chunk`-sized ranges from a shared counter.
pub const Chunks = struct {
    next: Atomic(usize) = .init(0),
    total: usize,
    chunk: usize,

    pub fn init(total: usize, chunk: usize) Chunks {
        return .{ .total = total, .chunk = @max(chunk, 1) };
    }

    pub fn grab(self: *Chunks) ?[2]usize {
        const s = self.next.fetchAdd(self.chunk, .monotonic);
        if (s >= self.total) return null;
        return .{ s, @min(s + self.chunk, self.total) };
    }
};

/// Number of physical cores (Linux SMT-aware), falling back to logical count.
pub fn defaultThreads(io: std.Io) usize {
    const logical = std.Thread.getCpuCount() catch 4;
    var buf: [8]u8 = undefined;
    const smt = std.Io.Dir.cwd().readFile(io, "/sys/devices/system/cpu/smt/active", &buf) catch return logical;
    if (smt.len > 0 and smt[0] == '1' and logical >= 2) return logical / 2;
    return logical;
}
