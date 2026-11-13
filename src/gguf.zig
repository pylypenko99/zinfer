//! Zero-copy GGUF reader. The file is memory-mapped; metadata strings and
//! tensor data are slices into the mapping.
const std = @import("std");
const posix = std.posix;

pub const GgmlType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    bf16 = 30,
    _,

    /// Elements per block and bytes per block; null for unsupported types.
    pub fn layout(t: GgmlType) ?struct { block: usize, bytes: usize } {
        return switch (t) {
            .f32 => .{ .block = 1, .bytes = 4 },
            .f16, .bf16 => .{ .block = 1, .bytes = 2 },
            .q4_0 => .{ .block = 32, .bytes = 18 },
            .q8_0 => .{ .block = 32, .bytes = 34 },
            .q4_k => .{ .block = 256, .bytes = 144 },
            .q5_k => .{ .block = 256, .bytes = 176 },
            .q6_k => .{ .block = 256, .bytes = 210 },
            else => null,
        };
    }

    pub fn rowBytes(t: GgmlType, n: usize) ?usize {
        const l = t.layout() orelse return null;
        if (n % l.block != 0) return null;
        return n / l.block * l.bytes;
    }
};

pub const ValueType = enum(u32) {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    u32 = 4,
    i32 = 5,
    f32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    u64 = 10,
    i64 = 11,
    f64 = 12,
    _,

    fn fixedSize(t: ValueType) ?usize {
        return switch (t) {
            .u8, .i8, .bool => 1,
            .u16, .i16 => 2,
            .u32, .i32, .f32 => 4,
            .u64, .i64, .f64 => 8,
            else => null,
        };
    }
};

pub const Array = struct {
    elem: ValueType,
    len: usize,
    /// Raw element bytes (for strings: length-prefixed sequence).
    data: []const u8,

    pub fn strings(a: Array) StringIter {
        std.debug.assert(a.elem == .string);
        return .{ .data = a.data };
    }

    pub fn item(a: Array, comptime T: type, i: usize) T {
        const sz = @sizeOf(T);
        return std.mem.readInt(T, a.data[i * sz ..][0..sz], .little);
    }

    pub fn f32At(a: Array, i: usize) f32 {
        return @bitCast(a.item(u32, i));
    }
};

pub const StringIter = struct {
    data: []const u8,
    pos: usize = 0,

    pub fn next(it: *StringIter) []const u8 {
        const n: usize = @intCast(std.mem.readInt(u64, it.data[it.pos..][0..8], .little));
        const s = it.data[it.pos + 8 ..][0..n];
        it.pos += 8 + n;
        return s;
    }
};

pub const Value = union(enum) {
    int: i64,
    uint: u64,
    float: f64,
    boolean: bool,
    string: []const u8,
    array: Array,
};

pub const TensorInfo = struct {
    name: []const u8,
    dims: [4]u64,
    n_dims: u32,
    typ: GgmlType,
    offset: u64,
    data: []const u8 = &.{},

    pub fn nelems(t: TensorInfo) u64 {
        var n: u64 = 1;
        for (t.dims[0..t.n_dims]) |d| n *= d;
        return n;
    }
};

pub const File = struct {
    map: []align(std.heap.page_size_min) const u8,
    kv: std.StringHashMapUnmanaged(Value) = .empty,
    tensors: std.StringHashMapUnmanaged(TensorInfo) = .empty,
    data_offset: usize = 0,
    version: u32 = 0,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !File {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        const size: usize = @intCast(try file.length(io));
        if (size < 24) return error.InvalidGguf;
        const map = try posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
        errdefer posix.munmap(map);
        var self: File = .{ .map = map };
        errdefer self.deinitTables(gpa);
        try self.parse(gpa);
        return self;
    }

    fn deinitTables(self: *File, gpa: std.mem.Allocator) void {
        self.kv.deinit(gpa);
        self.tensors.deinit(gpa);
    }

    pub fn deinit(self: *File, gpa: std.mem.Allocator) void {
        self.deinitTables(gpa);
        posix.munmap(self.map);
    }

    const Cursor = struct {
        buf: []const u8,
        pos: usize = 0,

        fn int(c: *Cursor, comptime T: type) !T {
            const sz = @sizeOf(T);
            if (c.pos + sz > c.buf.len) return error.InvalidGguf;
            const v = std.mem.readInt(T, c.buf[c.pos..][0..sz], .little);
            c.pos += sz;
            return v;
        }

        fn bytes(c: *Cursor, n: usize) ![]const u8 {
            if (n > c.buf.len - c.pos) return error.InvalidGguf;
            const s = c.buf[c.pos..][0..n];
            c.pos += n;
            return s;
        }

        fn string(c: *Cursor) ![]const u8 {
            return c.bytes(@intCast(try c.int(u64)));
        }

        fn value(c: *Cursor, t: ValueType) !Value {
            return switch (t) {
                .u8 => .{ .uint = try c.int(u8) },
                .u16 => .{ .uint = try c.int(u16) },
                .u32 => .{ .uint = try c.int(u32) },
                .u64 => .{ .uint = try c.int(u64) },
                .i8 => .{ .int = try c.int(i8) },
                .i16 => .{ .int = try c.int(i16) },
                .i32 => .{ .int = try c.int(i32) },
                .i64 => .{ .int = try c.int(i64) },
                .f32 => .{ .float = @as(f32, @bitCast(try c.int(u32))) },
                .f64 => .{ .float = @as(f64, @bitCast(try c.int(u64))) },
                .bool => .{ .boolean = (try c.int(u8)) != 0 },
                .string => .{ .string = try c.string() },
                .array => blk: {
                    const et: ValueType = @fromBackingInt(@intCast(try c.int(u32)));
                    const n: usize = @intCast(try c.int(u64));
                    const start = c.pos;
                    if (et.fixedSize()) |sz| {
                        _ = try c.bytes(n * sz);
                    } else if (et == .string) {
                        for (0..n) |_| _ = try c.string();
                    } else return error.UnsupportedGgufArray;
                    break :blk .{ .array = .{ .elem = et, .len = n, .data = c.buf[start..c.pos] } };
                },
                _ => error.InvalidGguf,
            };
        }
    };

    fn parse(self: *File, gpa: std.mem.Allocator) !void {
        var c: Cursor = .{ .buf = self.map };
        if (!std.mem.eql(u8, try c.bytes(4), "GGUF")) return error.NotGguf;
        self.version = try c.int(u32);
        if (self.version < 2) return error.UnsupportedGgufVersion;
        const n_tensors: usize = @intCast(try c.int(u64));
        const n_kv: usize = @intCast(try c.int(u64));

        try self.kv.ensureTotalCapacity(gpa, @intCast(n_kv));
        for (0..n_kv) |_| {
            const key = try c.string();
            const t: ValueType = @fromBackingInt(@intCast(try c.int(u32)));
            self.kv.putAssumeCapacity(key, try c.value(t));
        }

        try self.tensors.ensureTotalCapacity(gpa, @intCast(n_tensors));
        for (0..n_tensors) |_| {
            var ti: TensorInfo = .{ .name = try c.string(), .dims = .{ 1, 1, 1, 1 }, .n_dims = try c.int(u32), .typ = undefined, .offset = undefined };
            if (ti.n_dims > 4) return error.InvalidGguf;
            for (0..ti.n_dims) |d| ti.dims[d] = try c.int(u64);
            ti.typ = @fromBackingInt(@intCast(try c.int(u32)));
            ti.offset = try c.int(u64);
            self.tensors.putAssumeCapacity(ti.name, ti);
        }

        const alignment: usize = @intCast(self.getUint("general.alignment") orelse 32);
        self.data_offset = std.mem.alignForward(usize, c.pos, alignment);

        var it = self.tensors.valueIterator();
        while (it.next()) |ti| {
            const l = ti.typ.layout() orelse continue; // reported at use site
            const n: usize = @intCast(ti.nelems());
            const nbytes = n / l.block * l.bytes;
            const start = self.data_offset + @as(usize, @intCast(ti.offset));
            if (start + nbytes > self.map.len) return error.TensorOutOfBounds;
            ti.data = self.map[start..][0..nbytes];
        }
    }

    pub fn get(self: *const File, key: []const u8) ?Value {
        return self.kv.get(key);
    }

    pub fn getUint(self: *const File, key: []const u8) ?u64 {
        return switch (self.get(key) orelse return null) {
            .uint => |v| v,
            .int => |v| if (v >= 0) @intCast(v) else null,
            else => null,
        };
    }

    pub fn getFloat(self: *const File, key: []const u8) ?f64 {
        return switch (self.get(key) orelse return null) {
            .float => |v| v,
            .uint => |v| @floatFromInt(v),
            .int => |v| @floatFromInt(v),
            else => null,
        };
    }

    pub fn getBool(self: *const File, key: []const u8) ?bool {
        return switch (self.get(key) orelse return null) {
            .boolean => |v| v,
            else => null,
        };
    }

    pub fn getString(self: *const File, key: []const u8) ?[]const u8 {
        return switch (self.get(key) orelse return null) {
            .string => |v| v,
            else => null,
        };
    }

    pub fn getArray(self: *const File, key: []const u8) ?Array {
        return switch (self.get(key) orelse return null) {
            .array => |v| v,
            else => null,
        };
    }

    /// Byte range holding all tensor data.
    pub fn tensorRegion(self: *const File) []const u8 {
        return self.map[self.data_offset..];
    }
};
