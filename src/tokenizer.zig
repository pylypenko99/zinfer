//! Tokenizers read from GGUF metadata:
//!  - "gpt2": byte-level BPE with merge ranks (Llama 3, Qwen2/3, ...)
//!  - "llama": SentencePiece BPE with scores and byte fallback (Llama 2, Mistral, TinyLlama)
//! Special (control / user-defined) tokens in the input text are matched verbatim.
const std = @import("std");
const gguf = @import("gguf.zig");
const Allocator = std.mem.Allocator;

pub const Kind = enum { bpe, spm };

/// Pre-tokenizer split rules for byte-level BPE.
pub const Pre = enum {
    /// Llama 3 regex: digits grouped by up to 3.
    llama3,
    /// Qwen2 regex: identical to llama3 but single digits.
    qwen2,
    /// Original GPT-2 regex.
    gpt2,
};

const TokType = enum(u8) { undefined = 0, normal = 1, unknown = 2, control = 3, user_defined = 4, unused = 5, byte = 6, _ };

const Merge = struct { rank: u32, id: u32 };

pub const Tokenizer = struct {
    kind: Kind,
    pre: Pre = .llama3,
    ignore_merges: bool = false,
    vocab: []const []const u8,
    types: []TokType,
    scores: []f32 = &.{},
    map: std.StringHashMapUnmanaged(u32) = .empty,
    merges: std.AutoHashMapUnmanaged(u64, Merge) = .empty,
    /// Decoded bytes of every token.
    piece_off: []u32,
    piece_buf: []u8,
    /// Special tokens sorted by descending length.
    specials: []u32,
    special_first: std.StaticBitSet(256) = .empty,
    /// Token for each raw byte (gpt2 mapped char, or spm <0xXX>).
    byte_tok: [256]?u32 = @splat(null),
    bos: ?u32 = null,
    eos: ?u32 = null,
    add_bos: bool = false,
    add_space_prefix: bool = false,
    eog: []u32,

    pub fn init(gpa: Allocator, f: *const gguf.File) !Tokenizer {
        const model = f.getString("tokenizer.ggml.model") orelse return error.MissingTokenizer;
        const kind: Kind = if (std.mem.eql(u8, model, "gpt2"))
            .bpe
        else if (std.mem.eql(u8, model, "llama"))
            .spm
        else {
            std.log.err("unsupported tokenizer model '{s}'", .{model});
            return error.UnsupportedTokenizer;
        };

        const tok_arr = f.getArray("tokenizer.ggml.tokens") orelse return error.MissingTokenizer;
        const n = tok_arr.len;
        const vocab = try gpa.alloc([]const u8, n);
        errdefer gpa.free(vocab);
        var it = tok_arr.strings();
        for (vocab) |*v| v.* = it.next();

        const types = try gpa.alloc(TokType, n);
        errdefer gpa.free(types);
        @memset(types, .normal);
        if (f.getArray("tokenizer.ggml.token_type")) |ta| {
            for (types, 0..) |*t, i| t.* = @fromBackingInt(@intCast(@as(u8, @intCast(ta.item(i32, i)))));
        }

        var self: Tokenizer = .{
            .kind = kind,
            .vocab = vocab,
            .types = types,
            .piece_off = &.{},
            .piece_buf = &.{},
            .specials = &.{},
            .eog = &.{},
        };
        errdefer self.deinit(gpa);

        try self.map.ensureTotalCapacity(gpa, @intCast(n));
        // Iterate backwards so the lowest id wins on duplicate strings.
        var i = n;
        while (i > 0) {
            i -= 1;
            self.map.putAssumeCapacity(vocab[i], @intCast(i));
        }

        if (kind == .spm) {
            const sa = f.getArray("tokenizer.ggml.scores") orelse return error.MissingTokenizer;
            self.scores = try gpa.alloc(f32, n);
            for (self.scores, 0..) |*s, k| s.* = sa.f32At(k);
            for (0..256) |b| {
                var buf: [6]u8 = undefined;
                const s = std.fmt.bufPrint(&buf, "<0x{X:0>2}>", .{b}) catch unreachable;
                self.byte_tok[b] = self.map.get(s);
            }
            self.add_space_prefix = f.getBool("tokenizer.ggml.add_space_prefix") orelse true;
        } else {
            const pre = f.getString("tokenizer.ggml.pre") orelse "default";
            self.pre = if (std.mem.eql(u8, pre, "qwen2")) .qwen2 else if (std.mem.eql(u8, pre, "gpt-2") or std.mem.eql(u8, pre, "gpt2")) .gpt2 else .llama3;
            if (!std.mem.eql(u8, pre, "llama-bpe") and !std.mem.eql(u8, pre, "llama3") and self.pre == .llama3)
                std.log.warn("pre-tokenizer '{s}' not implemented exactly, using llama3 rules", .{pre});
            self.ignore_merges = std.mem.eql(u8, pre, "llama-bpe") or std.mem.eql(u8, pre, "llama3");
            for (0..256) |b| {
                var buf: [4]u8 = undefined;
                const len = std.unicode.utf8Encode(byte_to_cp[b], &buf) catch unreachable;
                self.byte_tok[b] = self.map.get(buf[0..len]);
            }
            const ma = f.getArray("tokenizer.ggml.merges") orelse return error.MissingTokenizer;
            try self.merges.ensureTotalCapacity(gpa, @intCast(ma.len));
            var mit = ma.strings();
            var cat: std.ArrayList(u8) = .empty;
            defer cat.deinit(gpa);
            for (0..ma.len) |rank| {
                const m = mit.next();
                const sp = std.mem.indexOfScalarPos(u8, m, 1, ' ') orelse continue;
                const a = self.map.get(m[0..sp]) orelse continue;
                const b = self.map.get(m[sp + 1 ..]) orelse continue;
                cat.clearRetainingCapacity();
                try cat.appendSlice(gpa, m[0..sp]);
                try cat.appendSlice(gpa, m[sp + 1 ..]);
                const id = self.map.get(cat.items) orelse continue;
                const gop = self.merges.getOrPutAssumeCapacity(pairKey(a, b));
                if (!gop.found_existing) gop.value_ptr.* = .{ .rank = @intCast(rank), .id = id };
            }
        }

        self.bos = if (f.getUint("tokenizer.ggml.bos_token_id")) |v| @intCast(v) else null;
        self.eos = if (f.getUint("tokenizer.ggml.eos_token_id")) |v| @intCast(v) else null;
        self.add_bos = f.getBool("tokenizer.ggml.add_bos_token") orelse (kind == .spm);

        try self.buildPieces(gpa);
        try self.buildSpecials(gpa);
        try self.buildEog(gpa, f);
        return self;
    }

    pub fn deinit(self: *Tokenizer, gpa: Allocator) void {
        gpa.free(self.vocab);
        gpa.free(self.types);
        gpa.free(self.scores);
        self.map.deinit(gpa);
        self.merges.deinit(gpa);
        gpa.free(self.piece_off);
        gpa.free(self.piece_buf);
        gpa.free(self.specials);
        gpa.free(self.eog);
    }

    fn pairKey(a: u32, b: u32) u64 {
        return (@as(u64, a) << 32) | b;
    }

    fn isSpecial(t: TokType) bool {
        return t == .control or t == .user_defined;
    }

    fn buildPieces(self: *Tokenizer, gpa: Allocator) !void {
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(gpa);
        const off = try gpa.alloc(u32, self.vocab.len + 1);
        errdefer gpa.free(off);
        for (self.vocab, 0..) |s, id| {
            off[id] = @intCast(buf.items.len);
            switch (self.types[id]) {
                .control, .unused, .unknown => {},
                .user_defined => try buf.appendSlice(gpa, s),
                .byte => if (self.kind == .spm and s.len == 6) {
                    try buf.append(gpa, std.fmt.parseInt(u8, s[3..5], 16) catch '?');
                } else try buf.appendSlice(gpa, s),
                else => switch (self.kind) {
                    .spm => {
                        var k: usize = 0;
                        while (k < s.len) {
                            if (std.mem.startsWith(u8, s[k..], "\u{2581}")) {
                                try buf.append(gpa, ' ');
                                k += 3;
                            } else {
                                try buf.append(gpa, s[k]);
                                k += 1;
                            }
                        }
                    },
                    .bpe => {
                        var view = std.unicode.Utf8View.initUnchecked(s);
                        var cit = view.iterator();
                        while (cit.nextCodepoint()) |cp| {
                            if (cp < cp_to_byte.len and cp_to_byte[cp] >= 0) {
                                try buf.append(gpa, @intCast(cp_to_byte[cp]));
                            } else {
                                var tmp: [4]u8 = undefined;
                                const l = std.unicode.utf8Encode(cp, &tmp) catch 0;
                                try buf.appendSlice(gpa, tmp[0..l]);
                            }
                        }
                    },
                },
            }
        }
        off[self.vocab.len] = @intCast(buf.items.len);
        self.piece_off = off;
        self.piece_buf = try buf.toOwnedSlice(gpa);
    }

    fn buildSpecials(self: *Tokenizer, gpa: Allocator) !void {
        var list: std.ArrayList(u32) = .empty;
        errdefer list.deinit(gpa);
        for (self.types, 0..) |t, id| {
            if (isSpecial(t) and self.vocab[id].len > 0) {
                try list.append(gpa, @intCast(id));
                self.special_first.set(self.vocab[id][0]);
            }
        }
        std.mem.sort(u32, list.items, self, struct {
            fn lt(s: *Tokenizer, a: u32, b: u32) bool {
                return s.vocab[a].len > s.vocab[b].len;
            }
        }.lt);
        self.specials = try list.toOwnedSlice(gpa);
    }

    fn buildEog(self: *Tokenizer, gpa: Allocator, f: *const gguf.File) !void {
        var list: std.ArrayList(u32) = .empty;
        errdefer list.deinit(gpa);
        if (self.eos) |e| try list.append(gpa, e);
        for ([_][]const u8{ "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id" }) |k| {
            if (f.getUint(k)) |v| try list.append(gpa, @intCast(v));
        }
        for ([_][]const u8{ "<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>", "</s>", "<end_of_turn>" }) |s| {
            if (self.map.get(s)) |id| if (isSpecial(self.types[id]) and std.mem.indexOfScalar(u32, list.items, id) == null)
                try list.append(gpa, id);
        }
        self.eog = try list.toOwnedSlice(gpa);
    }

    pub fn isEog(self: *const Tokenizer, id: u32) bool {
        return std.mem.indexOfScalar(u32, self.eog, id) != null;
    }

    pub fn piece(self: *const Tokenizer, id: u32) []const u8 {
        return self.piece_buf[self.piece_off[id]..self.piece_off[id + 1]];
    }

    pub fn tokenId(self: *const Tokenizer, s: []const u8) ?u32 {
        return self.map.get(s);
    }

    /// Encode text. Special-token strings are matched when `parse_special`.
    pub fn encode(self: *const Tokenizer, gpa: Allocator, text: []const u8, add_bos: bool, parse_special: bool, out: *std.ArrayList(u32)) !void {
        if (add_bos and self.add_bos) if (self.bos) |b| try out.append(gpa, b);
        var start: usize = 0;
        var i: usize = 0;
        var first_fragment = true;
        while (i < text.len) {
            if (parse_special and self.special_first.isSet(text[i])) {
                if (self.matchSpecial(text[i..])) |id| {
                    if (i > start) try self.encodeText(gpa, text[start..i], first_fragment, out);
                    first_fragment = false;
                    try out.append(gpa, id);
                    i += self.vocab[id].len;
                    start = i;
                    continue;
                }
            }
            i += 1;
        }
        if (text.len > start) try self.encodeText(gpa, text[start..], first_fragment, out);
    }

    fn matchSpecial(self: *const Tokenizer, s: []const u8) ?u32 {
        for (self.specials) |id| if (std.mem.startsWith(u8, s, self.vocab[id])) return id;
        return null;
    }

    fn encodeText(self: *const Tokenizer, gpa: Allocator, text: []const u8, first: bool, out: *std.ArrayList(u32)) !void {
        switch (self.kind) {
            .bpe => try self.encodeBpe(gpa, text, out),
            .spm => try self.encodeSpm(gpa, text, first, out),
        }
    }

    // ----------------------------------------------------------------- BPE

    fn encodeBpe(self: *const Tokenizer, gpa: Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        var cps: std.ArrayList(Cp) = .empty;
        defer cps.deinit(gpa);
        try decodeCps(gpa, text, &cps);
        var syms: std.ArrayList(u32) = .empty;
        defer syms.deinit(gpa);
        var mapped: std.ArrayList(u8) = .empty;
        defer mapped.deinit(gpa);

        var i: usize = 0;
        while (i < cps.items.len) {
            const j = splitAt(self.pre, cps.items, i);
            const a = cps.items[i].off;
            const b = if (j < cps.items.len) cps.items[j].off else text.len;
            try self.bpeWord(gpa, text[a..b], &syms, &mapped, out);
            i = j;
        }
    }

    fn bpeWord(self: *const Tokenizer, gpa: Allocator, word: []const u8, syms: *std.ArrayList(u32), mapped: *std.ArrayList(u8), out: *std.ArrayList(u32)) !void {
        if (self.ignore_merges) {
            mapped.clearRetainingCapacity();
            for (word) |c| {
                var tmp: [4]u8 = undefined;
                const l = std.unicode.utf8Encode(byte_to_cp[c], &tmp) catch unreachable;
                try mapped.appendSlice(gpa, tmp[0..l]);
            }
            if (self.map.get(mapped.items)) |id| return out.append(gpa, id);
        }
        syms.clearRetainingCapacity();
        for (word) |c| if (self.byte_tok[c]) |t| try syms.append(gpa, t);
        while (syms.items.len > 1) {
            var best: ?Merge = null;
            var at: usize = 0;
            for (0..syms.items.len - 1) |k| {
                if (self.merges.get(pairKey(syms.items[k], syms.items[k + 1]))) |m| {
                    if (best == null or m.rank < best.?.rank) {
                        best = m;
                        at = k;
                    }
                }
            }
            const m = best orelse break;
            syms.items[at] = m.id;
            _ = syms.orderedRemove(at + 1);
        }
        try out.appendSlice(gpa, syms.items);
    }

    // ----------------------------------------------------------------- SPM

    fn encodeSpm(self: *const Tokenizer, gpa: Allocator, text: []const u8, first: bool, out: *std.ArrayList(u32)) !void {
        var norm: std.ArrayList(u8) = .empty;
        defer norm.deinit(gpa);
        if (first and self.add_space_prefix) try norm.appendSlice(gpa, "\u{2581}");
        for (text) |c| {
            if (c == ' ') try norm.appendSlice(gpa, "\u{2581}") else try norm.append(gpa, c);
        }
        const s = norm.items;
        const Sym = struct { a: usize, b: usize };
        var syms: std.ArrayList(Sym) = .empty;
        defer syms.deinit(gpa);
        var k: usize = 0;
        while (k < s.len) {
            const l = @min(std.unicode.utf8ByteSequenceLength(s[k]) catch 1, s.len - k);
            try syms.append(gpa, .{ .a = k, .b = k + l });
            k += l;
        }
        while (syms.items.len > 1) {
            var best_score: f32 = -std.math.inf(f32);
            var best_at: ?usize = null;
            for (0..syms.items.len - 1) |q| {
                if (self.map.get(s[syms.items[q].a..syms.items[q + 1].b])) |id| {
                    if (self.scores[id] > best_score) {
                        best_score = self.scores[id];
                        best_at = q;
                    }
                }
            }
            const q = best_at orelse break;
            syms.items[q].b = syms.items[q + 1].b;
            _ = syms.orderedRemove(q + 1);
        }
        for (syms.items) |sym| {
            if (self.map.get(s[sym.a..sym.b])) |id| {
                try out.append(gpa, id);
            } else for (s[sym.a..sym.b]) |c| {
                if (self.byte_tok[c]) |t| try out.append(gpa, t);
            }
        }
    }
};

// --------------------------------------------------------------------------
// GPT-2 byte <-> unicode mapping

const byte_to_cp: [256]u21 = blk: {
    var m: [256]u21 = undefined;
    var n: u21 = 0;
    for (0..256) |b| {
        if ((b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255)) {
            m[b] = b;
        } else {
            m[b] = 256 + n;
            n += 1;
        }
    }
    break :blk m;
};

const cp_to_byte: [324]i16 = blk: {
    var m: [324]i16 = @splat(-1);
    for (byte_to_cp, 0..) |cp, b| m[cp] = b;
    break :blk m;
};

// --------------------------------------------------------------------------
// Pre-tokenizer (hand-written equivalent of the split regexes)

const Cp = struct { cp: u21, off: usize };

fn decodeCps(gpa: Allocator, text: []const u8, out: *std.ArrayList(Cp)) !void {
    var i: usize = 0;
    while (i < text.len) {
        const l = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
        const cp: u21 = if (i + l <= text.len)
            std.unicode.utf8Decode(text[i..][0..l]) catch 0xFFFD
        else
            0xFFFD;
        try out.append(gpa, .{ .cp = cp, .off = i });
        i += if (i + l <= text.len) l else 1;
    }
}

const Class = enum { letter, number, space, other };

fn inRanges(cp: u21, comptime ranges: []const [2]u21) bool {
    for (ranges) |r| if (cp >= r[0] and cp <= r[1]) return true;
    return false;
}

/// Approximate Unicode general-category classes (\p{L}, \p{N}, \s, rest).
/// ASCII is exact; non-ASCII covers common punctuation, symbols, digits,
/// combining marks and spaces; everything else counts as a letter.
pub fn classify(cp: u21) Class {
    if (cp < 128) {
        const c: u8 = @intCast(cp);
        if (std.ascii.isAlphabetic(c)) return .letter;
        if (std.ascii.isDigit(c)) return .number;
        if (c == ' ' or (c >= 9 and c <= 13)) return .space;
        return .other;
    }
    if (inRanges(cp, &.{ .{ 0x85, 0x85 }, .{ 0xA0, 0xA0 }, .{ 0x1680, 0x1680 }, .{ 0x2000, 0x200A }, .{ 0x2028, 0x2029 }, .{ 0x202F, 0x202F }, .{ 0x205F, 0x205F }, .{ 0x3000, 0x3000 } })) return .space;
    if (inRanges(cp, &.{
        .{ 0xB2, 0xB3 },     .{ 0xB9, 0xB9 },     .{ 0xBC, 0xBE },     .{ 0x660, 0x669 },     .{ 0x6F0, 0x6F9 },
        .{ 0x7C0, 0x7C9 },   .{ 0x966, 0x96F },   .{ 0x9E6, 0x9EF },   .{ 0xA66, 0xA6F },     .{ 0xAE6, 0xAEF },
        .{ 0xB66, 0xB6F },   .{ 0xBE6, 0xBF2 },   .{ 0xC66, 0xC6F },   .{ 0xCE6, 0xCEF },     .{ 0xD66, 0xD78 },
        .{ 0xE50, 0xE59 },   .{ 0xED0, 0xED9 },   .{ 0xF20, 0xF33 },   .{ 0x1040, 0x1049 },   .{ 0x17E0, 0x17E9 },
        .{ 0x1810, 0x1819 }, .{ 0x2070, 0x2070 }, .{ 0x2074, 0x2079 }, .{ 0x2080, 0x2089 },   .{ 0x2150, 0x2189 },
        .{ 0x2460, 0x249B }, .{ 0x24EA, 0x24FF }, .{ 0x2776, 0x2793 }, .{ 0x3007, 0x3007 },   .{ 0x3021, 0x3029 },
        .{ 0x3038, 0x303A }, .{ 0x3192, 0x3195 }, .{ 0x3220, 0x3229 }, .{ 0x3248, 0x324F },   .{ 0x3251, 0x325F },
        .{ 0x3280, 0x3289 }, .{ 0x32B1, 0x32BF }, .{ 0xFF10, 0xFF19 }, .{ 0x1D7CE, 0x1D7FF }, .{ 0x1F100, 0x1F10C },
    })) return .number;
    if (inRanges(cp, &.{
        // Latin-1 punctuation & symbols (letters ª µ º excluded)
        .{ 0xA1, 0xA9 },       .{ 0xAB, 0xB1 },        .{ 0xB4, 0xB4 },       .{ 0xB6, 0xB8 },       .{ 0xBB, 0xBB },
        .{ 0xBF, 0xBF },       .{ 0xD7, 0xD7 },        .{ 0xF7, 0xF7 },
        // Modifier symbols, combining marks
              .{ 0x2C2, 0x2C5 },     .{ 0x2D2, 0x2DF },
        .{ 0x2E5, 0x2EB },     .{ 0x2ED, 0x2ED },      .{ 0x2EF, 0x36F },     .{ 0x375, 0x375 },     .{ 0x37E, 0x37E },
        .{ 0x384, 0x385 },     .{ 0x387, 0x387 },      .{ 0x3F6, 0x3F6 },     .{ 0x482, 0x489 },     .{ 0x55A, 0x55F },
        .{ 0x589, 0x58A },     .{ 0x58D, 0x58F },      .{ 0x591, 0x5C7 },     .{ 0x5F3, 0x5F4 },     .{ 0x600, 0x61F },
        .{ 0x64B, 0x65F },     .{ 0x66A, 0x66D },      .{ 0x670, 0x670 },     .{ 0x6D4, 0x6D4 },     .{ 0x6D6, 0x6ED },
        .{ 0x6FD, 0x6FE },     .{ 0x700, 0x70F },      .{ 0x711, 0x711 },     .{ 0x730, 0x74A },     .{ 0x7A6, 0x7B0 },
        .{ 0x900, 0x903 },     .{ 0x93A, 0x93C },      .{ 0x93E, 0x94F },     .{ 0x951, 0x957 },     .{ 0x962, 0x965 },
        .{ 0x970, 0x970 },     .{ 0x981, 0x983 },      .{ 0x9BC, 0x9BC },     .{ 0x9BE, 0x9CD },     .{ 0x9D7, 0x9D7 },
        .{ 0x9E2, 0x9E3 },     .{ 0xA01, 0xA03 },      .{ 0xA3C, 0xA51 },     .{ 0xA70, 0xA71 },     .{ 0xA81, 0xA83 },
        .{ 0xABC, 0xABC },     .{ 0xABE, 0xACD },      .{ 0xB01, 0xB03 },     .{ 0xB3C, 0xB3C },     .{ 0xB3E, 0xB57 },
        .{ 0xBBE, 0xBCD },     .{ 0xC00, 0xC04 },      .{ 0xC3E, 0xC56 },     .{ 0xC81, 0xC83 },     .{ 0xCBC, 0xCBC },
        .{ 0xCBE, 0xCD6 },     .{ 0xD00, 0xD03 },      .{ 0xD3E, 0xD57 },     .{ 0xE31, 0xE31 },     .{ 0xE34, 0xE3A },
        .{ 0xE3F, 0xE3F },     .{ 0xE47, 0xE4F },      .{ 0xEB1, 0xEB1 },     .{ 0xEB4, 0xEBC },     .{ 0xEC8, 0xECE },
        .{ 0xF01, 0xF17 },     .{ 0xF3A, 0xF3F },      .{ 0xF71, 0xF87 },     .{ 0x102B, 0x103E },   .{ 0x104A, 0x104F },
        .{ 0x10FB, 0x10FB },   .{ 0x1360, 0x1368 },    .{ 0x16EB, 0x16ED },   .{ 0x17B4, 0x17D6 },   .{ 0x1800, 0x180F },
        .{ 0x1AB0, 0x1AFF },   .{ 0x1DC0, 0x1DFF },    .{ 0x1FBD, 0x1FBD },   .{ 0x1FBF, 0x1FC1 },   .{ 0x1FCD, 0x1FCF },
        .{ 0x1FDD, 0x1FDF },   .{ 0x1FED, 0x1FEF },    .{ 0x1FFD, 0x1FFE },
        // General punctuation, super/subscript signs, currency, combining for symbols
          .{ 0x200B, 0x2027 },   .{ 0x2030, 0x205E },
        .{ 0x2060, 0x206F },   .{ 0x207A, 0x207E },    .{ 0x208A, 0x208E },   .{ 0x20A0, 0x20FF },
        // Letterlike symbols (mixed; mostly symbols), arrows ... misc symbols, dingbats
          .{ 0x2100, 0x2101 },
        .{ 0x2103, 0x2106 },   .{ 0x2108, 0x2109 },    .{ 0x2114, 0x2114 },   .{ 0x2116, 0x2118 },   .{ 0x211E, 0x2123 },
        .{ 0x2125, 0x2125 },   .{ 0x2127, 0x2127 },    .{ 0x2129, 0x2129 },   .{ 0x212E, 0x212E },   .{ 0x213A, 0x213B },
        .{ 0x2140, 0x2144 },   .{ 0x214A, 0x214D },    .{ 0x214F, 0x214F },   .{ 0x218A, 0x245F },   .{ 0x249C, 0x24E9 },
        .{ 0x2500, 0x2775 },   .{ 0x2794, 0x2BFF },    .{ 0x2CE5, 0x2CEA },   .{ 0x2CF9, 0x2CFF },   .{ 0x2DE0, 0x2E7F },
        .{ 0x2E80, 0x2FFF },   .{ 0x3001, 0x3004 },    .{ 0x3008, 0x3020 },   .{ 0x302A, 0x3030 },   .{ 0x3036, 0x3037 },
        .{ 0x303D, 0x303F },   .{ 0x3099, 0x309C },    .{ 0x30A0, 0x30A0 },   .{ 0x30FB, 0x30FB },   .{ 0x3190, 0x3191 },
        .{ 0x3196, 0x319F },   .{ 0x31C0, 0x31E3 },    .{ 0x3200, 0x321E },   .{ 0x322A, 0x3247 },   .{ 0x3250, 0x3250 },
        .{ 0x3260, 0x327F },   .{ 0x328A, 0x32B0 },    .{ 0x32C0, 0x33FF },   .{ 0x4DC0, 0x4DFF },   .{ 0xA490, 0xA4C6 },
        .{ 0xA4FE, 0xA4FF },   .{ 0xA60D, 0xA60F },    .{ 0xA66F, 0xA67E },   .{ 0xA69E, 0xA69F },   .{ 0xA6F0, 0xA6F7 },
        .{ 0xA700, 0xA716 },   .{ 0xA720, 0xA721 },    .{ 0xA789, 0xA78A },   .{ 0xA802, 0xA802 },   .{ 0xA806, 0xA806 },
        .{ 0xA80B, 0xA80B },   .{ 0xA823, 0xA82B },    .{ 0xA874, 0xA877 },   .{ 0xA880, 0xA881 },   .{ 0xA8B4, 0xA8C5 },
        .{ 0xA8CE, 0xA8CF },   .{ 0xA8E0, 0xA8F1 },    .{ 0xA8F8, 0xA8FA },   .{ 0xA8FC, 0xA8FC },   .{ 0xE000, 0xF8FF },
        .{ 0xFB1E, 0xFB1E },   .{ 0xFB29, 0xFB29 },    .{ 0xFD3E, 0xFD3F },   .{ 0xFDFC, 0xFDFF },   .{ 0xFE00, 0xFE6F },
        .{ 0xFEFF, 0xFEFF },   .{ 0xFF01, 0xFF0F },    .{ 0xFF1A, 0xFF20 },   .{ 0xFF3B, 0xFF40 },   .{ 0xFF5B, 0xFF65 },
        .{ 0xFFE0, 0xFFFD },   .{ 0x1D000, 0x1D245 },  .{ 0x1D360, 0x1D378 }, .{ 0x1F000, 0x1F0FF }, .{ 0x1F10D, 0x1FAFF },
        .{ 0xE0000, 0xE01EF }, .{ 0xF0000, 0x10FFFF },
    })) return .other;
    return .letter;
}

fn isNl(cp: u21) bool {
    return cp == '\r' or cp == '\n';
}

fn lower(cp: u21) u21 {
    return if (cp >= 'A' and cp <= 'Z') cp + 32 else cp;
}

/// End (exclusive) of the pre-token starting at `i`.
fn splitAt(pre: Pre, cps: []const Cp, i: usize) usize {
    const n = cps.len;
    const cls = struct {
        fn f(s: []const Cp, k: usize) Class {
            return classify(s[k].cp);
        }
    }.f;
    const c = cps[i].cp;
    const ci = cls(cps, i);

    // 'contractions (case-insensitive except for gpt2)
    if (c == '\'' and i + 1 < n) {
        const fold = pre != .gpt2;
        const c1 = if (fold) lower(cps[i + 1].cp) else cps[i + 1].cp;
        if (c1 == 's' or c1 == 't' or c1 == 'm' or c1 == 'd') return i + 2;
        if (i + 2 < n) {
            const c2 = if (fold) lower(cps[i + 2].cp) else cps[i + 2].cp;
            if ((c1 == 'r' and c2 == 'e') or (c1 == 'v' and c2 == 'e') or (c1 == 'l' and c2 == 'l')) return i + 3;
        }
    }

    if (pre == .gpt2) {
        // ' ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+'
        var j = i;
        if (c == ' ' and i + 1 < n and cls(cps, i + 1) != .space) j += 1;
        const k = cls(cps, j);
        if (k != .space) {
            j += 1;
            while (j < n and cls(cps, j) == k) j += 1;
            return j;
        }
    } else {
        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (ci == .letter or (!isNl(c) and ci != .number and i + 1 < n and cls(cps, i + 1) == .letter)) {
            var j = i + 1;
            while (j < n and cls(cps, j) == .letter) j += 1;
            return j;
        }
        // \p{N}{1,3} (llama3) or \p{N} (qwen2)
        if (ci == .number) {
            const max: usize = if (pre == .qwen2) 1 else 3;
            var j = i + 1;
            while (j < n and j - i < max and cls(cps, j) == .number) j += 1;
            return j;
        }
        // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
        var j = i;
        if (c == ' ') j += 1;
        if (j < n and cls(cps, j) == .other) {
            j += 1;
            while (j < n and cls(cps, j) == .other) j += 1;
            while (j < n and isNl(cps[j].cp)) j += 1;
            return j;
        }
    }

    if (ci == .space) {
        var j = i;
        var last_nl: ?usize = null;
        while (j < n and cls(cps, j) == .space) : (j += 1) {
            if (isNl(cps[j].cp)) last_nl = j;
        }
        // \s*[\r\n]+
        if (pre != .gpt2) if (last_nl) |l| return l + 1;
        // \s+(?!\S) then \s+
        if (j == n or j - i == 1) return j;
        return j - 1;
    }
    return i + 1;
}

test "llama3 pre-tokenizer splits like the reference regex" {
    const gpa = std.testing.allocator;
    const cases = [_]struct { in: []const u8, out: []const []const u8 }{
        .{ .in = "Hello world", .out = &.{ "Hello", " world" } },
        .{ .in = "I'm  12345 ok!!\n\n x", .out = &.{ "I", "'m", " ", " ", "123", "45", " ok", "!!\n\n", " x" } },
        .{ .in = "a  \n  b", .out = &.{ "a", "  \n", " ", " b" } },
        .{ .in = "(hello) Привіт, світе", .out = &.{ "(hello", ")", " Привіт", ",", " світе" } },
        .{ .in = "x   ", .out = &.{ "x", "   " } },
    };
    for (cases) |cs| {
        var cps: std.ArrayList(Cp) = .empty;
        defer cps.deinit(gpa);
        try decodeCps(gpa, cs.in, &cps);
        var got: std.ArrayList([]const u8) = .empty;
        defer got.deinit(gpa);
        var i: usize = 0;
        while (i < cps.items.len) {
            const j = splitAt(.llama3, cps.items, i);
            const a = cps.items[i].off;
            const b = if (j < cps.items.len) cps.items[j].off else cs.in.len;
            try got.append(gpa, cs.in[a..b]);
            i = j;
        }
        try std.testing.expectEqual(cs.out.len, got.items.len);
        for (cs.out, got.items) |e, g| try std.testing.expectEqualStrings(e, g);
    }
}
