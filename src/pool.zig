//! Persistent spin-then-sleep thread pool. `run` broadcasts one job to every
//! worker (the caller participates as worker 0) and returns when all finish.
//! Workers spin between dispatches so back-to-back ops in a forward pass cost
//! no syscalls; after a short idle period they park on a futex.
const std = @import("std");
const builtin = @import("builtin");
const Atomic = std.atomic.Value;

const spin_limit = 1 << 16;
pub const max_discovered_cpus = 256;

const Job = struct {
    ctx: *anyopaque,
    func: *const fn (ctx: *anyopaque, tid: usize, nth: usize) void,
};

pub const CpuTopology = struct {
    cpus: [max_discovered_cpus]usize = undefined,
    count: usize = 0,
    physical_count: usize = 0,
};

/// Discovers CPU topology on Linux via sysfs, prioritizing distinct physical cores
/// before filling with SMT/hyperthreading siblings.
pub fn discoverTopology(io: std.Io) CpuTopology {
    var topo: CpuTopology = .{};
    const logical = std.Thread.getCpuCount() catch 4;
    const n = @min(logical, max_discovered_cpus);

    if (builtin.os.tag != .linux) {
        for (0..n) |i| topo.cpus[i] = i;
        topo.count = n;
        topo.physical_count = n;
        return topo;
    }

    var primary: [max_discovered_cpus]usize = undefined;
    var primary_len: usize = 0;
    var secondary: [max_discovered_cpus]usize = undefined;
    var secondary_len: usize = 0;

    var seen_cores: [max_discovered_cpus]u32 = undefined;
    var seen_len: usize = 0;

    for (0..n) |cpu| {
        var path_buf: [64]u8 = undefined;
        const core_path = std.fmt.bufPrint(&path_buf, "/sys/devices/system/cpu/cpu{d}/topology/core_id", .{cpu}) catch continue;
        var buf: [16]u8 = undefined;
        const core_str = std.Io.Dir.cwd().readFile(io, core_path, &buf) catch {
            primary[primary_len] = cpu;
            primary_len += 1;
            continue;
        };
        const core_id = std.fmt.parseInt(u16, std.mem.trim(u8, core_str, " \r\n\t"), 10) catch {
            primary[primary_len] = cpu;
            primary_len += 1;
            continue;
        };

        const pkg_path = std.fmt.bufPrint(&path_buf, "/sys/devices/system/cpu/cpu{d}/topology/physical_package_id", .{cpu}) catch continue;
        const pkg_str = std.Io.Dir.cwd().readFile(io, pkg_path, &buf) catch "0";
        const pkg_id = std.fmt.parseInt(u16, std.mem.trim(u8, pkg_str, " \r\n\t"), 10) catch 0;

        const key: u32 = (@as(u32, pkg_id) << 16) | @as(u32, core_id);

        var is_seen = false;
        for (seen_cores[0..seen_len]) |k| {
            if (k == key) {
                is_seen = true;
                break;
            }
        }

        if (!is_seen) {
            seen_cores[seen_len] = key;
            seen_len += 1;
            primary[primary_len] = cpu;
            primary_len += 1;
        } else {
            secondary[secondary_len] = cpu;
            secondary_len += 1;
        }
    }

    if (primary_len == 0) {
        for (0..n) |i| topo.cpus[i] = i;
        topo.count = n;
        topo.physical_count = n;
        return topo;
    }

    topo.physical_count = primary_len;
    var total: usize = 0;
    for (primary[0..primary_len]) |c| {
        topo.cpus[total] = c;
        total += 1;
    }
    for (secondary[0..secondary_len]) |c| {
        topo.cpus[total] = c;
        total += 1;
    }
    topo.count = total;
    return topo;
}

pub fn pinCurrentThread(cpu: usize) void {
    if (builtin.os.tag != .linux) return;
    var set: std.os.linux.cpu_set_t = undefined;
    @memset(&set, 0);
    const bits_per_word = @sizeOf(usize) * 8;
    const word_idx = cpu / bits_per_word;
    const bit_idx = cpu % bits_per_word;
    if (word_idx < set.len) {
        set[word_idx] |= (@as(usize, 1) << @intCast(bit_idx));
        std.os.linux.sched_setaffinity(0, &set) catch {};
    }
}

pub const Pool = struct {
    io: std.Io,
    threads: []std.Thread,
    nth: usize,
    job: Job = undefined,
    gen: Atomic(u32) = .init(0),
    done: Atomic(u32) = .init(0),
    sleepers: Atomic(u32) = .init(0),
    quit: bool = false,
    cpu_map: []usize,

    pub fn create(gpa: std.mem.Allocator, io: std.Io, nth: usize, pin: bool) !*Pool {
        const self = try gpa.create(Pool);
        errdefer gpa.destroy(self);
        self.* = .{
            .io = io,
            .threads = try gpa.alloc(std.Thread, nth - 1),
            .nth = nth,
            .cpu_map = try gpa.alloc(usize, nth),
        };
        errdefer gpa.free(self.threads);
        errdefer gpa.free(self.cpu_map);

        const topo = discoverTopology(io);
        for (0..nth) |i| {
            self.cpu_map[i] = if (topo.count > 0) topo.cpus[i % topo.count] else i;
        }

        if (pin) pinCurrentThread(self.cpu_map[0]);

        var spawned: usize = 0;
        errdefer {
            self.shutdown(spawned);
            gpa.free(self.threads);
            gpa.free(self.cpu_map);
        }
        while (spawned < nth - 1) : (spawned += 1) {
            const tid = spawned + 1;
            self.threads[spawned] = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, worker, .{ self, tid, pin });
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
        gpa.free(self.cpu_map);
        gpa.destroy(self);
    }

    fn worker(self: *Pool, tid: usize, pin: bool) void {
        if (pin) pinCurrentThread(self.cpu_map[tid]);
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

/// Number of physical cores (Linux topology-aware), falling back to logical count.
pub fn defaultThreads(io: std.Io) usize {
    const topo = discoverTopology(io);
    return if (topo.physical_count > 0) topo.physical_count else std.Thread.getCpuCount() catch 4;
}

test "discoverTopology" {
    const topo = discoverTopology(std.testing.io);
    try std.testing.expect(topo.count >= 1);
    try std.testing.expect(topo.physical_count >= 1);
    try std.testing.expect(topo.physical_count <= topo.count);

    // Verify all CPUs in the topology are unique
    for (0..topo.count) |i| {
        for (i + 1..topo.count) |j| {
            try std.testing.expect(topo.cpus[i] != topo.cpus[j]);
        }
    }
}

test "pinCurrentThread" {
    if (builtin.os.tag == .linux) {
        pinCurrentThread(0);
        var set: std.os.linux.cpu_set_t = undefined;
        @memset(&set, 0);
        const rc = std.os.linux.sched_getaffinity(0, @sizeOf(std.os.linux.cpu_set_t), &set);
        try std.testing.expectEqual(@as(usize, 0), rc);
        try std.testing.expect((set[0] & 1) != 0);
    }
}

test "Pool execution and pinning" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = try Pool.create(gpa, io, 2, true);
    defer pool.destroy(gpa);

    var res = [_]usize{ 0, 0 };
    const JobCtx = struct {
        r: *[2]usize,
    };
    var job_ctx = JobCtx{ .r = &res };
    pool.run(&job_ctx, struct {
        fn f(ctx: *JobCtx, tid: usize, _: usize) void {
            ctx.r[tid] = tid + 10;
        }
    }.f);

    try std.testing.expectEqual(@as(usize, 10), res[0]);
    try std.testing.expectEqual(@as(usize, 11), res[1]);
}
