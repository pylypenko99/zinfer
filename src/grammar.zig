//! Zero-allocation JSON Schema and Grammar-Constrained Logit Masking for zinfer.
//! Enforces valid JSON syntax (objects, arrays, strings, numbers, booleans, null)
//! directly during token sampling by setting logits of invalid tokens to -infinity.

const std = @import("std");
const Tokenizer = @import("tokenizer.zig").Tokenizer;

pub const ContainerType = enum {
    object,
    array,
};

pub const State = enum {
    start, // Waiting for '{' (or '[' if root is array)
    obj_expect_key_or_close, // Expecting '"' key or '}'
    inside_key, // Inside key string until unescaped '"'
    obj_expect_colon, // Expecting ':'
    expect_value, // Expecting any value: '"', '{', '[', number, true, false, null
    inside_str_value, // Inside string value until unescaped '"'
    inside_num_value, // Inside number
    inside_literal, // Inside true, false, null
    expect_comma_or_close, // Expecting ',' or '}' / ']'
    finished, // Root container closed, only whitespace / EOS allowed
};

pub const JsonValidator = struct {
    state: State = .start,
    depth: usize = 0,
    stack: [64]ContainerType = undefined,
    in_escape: bool = false,
    lit_expected: [8]u8 = undefined,
    lit_len: u8 = 0,
    lit_idx: u8 = 0,
    is_object_only: bool = true, // If true, root must be an object '{'

    pub fn initObjectOnly() JsonValidator {
        return .{ .is_object_only = true };
    }

    pub fn initAny() JsonValidator {
        return .{ .is_object_only = false };
    }

    pub fn isFinished(self: *const JsonValidator) bool {
        return self.state == .finished and self.depth == 0;
    }

    /// Feeds a single byte into the state machine.
    /// Returns false if the byte violates the grammar.
    pub fn stepByte(self: *JsonValidator, b: u8) bool {
        switch (self.state) {
            .start => {
                if (b == ' ' or b == '\t' or b == '\n' or b == '\r') return true;
                if (b == '{') {
                    if (self.depth >= self.stack.len) return false;
                    self.stack[self.depth] = .object;
                    self.depth += 1;
                    self.state = .obj_expect_key_or_close;
                    return true;
                }
                if (!self.is_object_only and b == '[') {
                    if (self.depth >= self.stack.len) return false;
                    self.stack[self.depth] = .array;
                    self.depth += 1;
                    self.state = .expect_value;
                    return true;
                }
                return false;
            },

            .obj_expect_key_or_close => {
                if (b == ' ' or b == '\t' or b == '\n' or b == '\r') return true;
                if (b == '}') {
                    if (self.depth == 0) return false;
                    self.depth -= 1;
                    self.state = if (self.depth == 0) .finished else .expect_comma_or_close;
                    return true;
                }
                if (b == '"') {
                    self.in_escape = false;
                    self.state = .inside_key;
                    return true;
                }
                return false;
            },

            .inside_key => {
                if (self.in_escape) {
                    self.in_escape = false;
                    return true;
                }
                if (b == '\\') {
                    self.in_escape = true;
                    return true;
                }
                if (b == '"') {
                    self.state = .obj_expect_colon;
                    return true;
                }
                // Control characters < 0x20 in JSON strings must be escaped
                if (b < 0x20) return false;
                return true;
            },

            .obj_expect_colon => {
                if (b == ' ' or b == '\t' or b == '\n' or b == '\r') return true;
                if (b == ':') {
                    self.state = .expect_value;
                    return true;
                }
                return false;
            },

            .expect_value => {
                if (b == ' ' or b == '\t' or b == '\n' or b == '\r') return true;
                if (self.depth > 0 and self.stack[self.depth - 1] == .array and b == ']') {
                    // Empty array close
                    self.depth -= 1;
                    self.state = if (self.depth == 0) .finished else .expect_comma_or_close;
                    return true;
                }
                if (b == '"') {
                    self.in_escape = false;
                    self.state = .inside_str_value;
                    return true;
                }
                if (b == '{') {
                    if (self.depth >= self.stack.len) return false;
                    self.stack[self.depth] = .object;
                    self.depth += 1;
                    self.state = .obj_expect_key_or_close;
                    return true;
                }
                if (b == '[') {
                    if (self.depth >= self.stack.len) return false;
                    self.stack[self.depth] = .array;
                    self.depth += 1;
                    self.state = .expect_value;
                    return true;
                }
                if (b == '-' or (b >= '0' and b <= '9')) {
                    self.state = .inside_num_value;
                    return true;
                }
                if (b == 't') {
                    // "true"
                    @memcpy(self.lit_expected[0..4], "true");
                    self.lit_len = 4;
                    self.lit_idx = 1;
                    self.state = .inside_literal;
                    return true;
                }
                if (b == 'f') {
                    // "false"
                    @memcpy(self.lit_expected[0..5], "false");
                    self.lit_len = 5;
                    self.lit_idx = 1;
                    self.state = .inside_literal;
                    return true;
                }
                if (b == 'n') {
                    // "null"
                    @memcpy(self.lit_expected[0..4], "null");
                    self.lit_len = 4;
                    self.lit_idx = 1;
                    self.state = .inside_literal;
                    return true;
                }
                return false;
            },

            .inside_str_value => {
                if (self.in_escape) {
                    self.in_escape = false;
                    return true;
                }
                if (b == '\\') {
                    self.in_escape = true;
                    return true;
                }
                if (b == '"') {
                    self.state = .expect_comma_or_close;
                    return true;
                }
                if (b < 0x20) return false;
                return true;
            },

            .inside_num_value => {
                if ((b >= '0' and b <= '9') or b == '.' or b == 'e' or b == 'E' or b == '+' or b == '-') {
                    return true;
                }
                // Number terminated by comma, close, or whitespace
                if (b == ' ' or b == '\t' or b == '\n' or b == '\r') {
                    self.state = .expect_comma_or_close;
                    return true;
                }
                if (b == ',') {
                    return self.transitionComma();
                }
                if (b == '}' or b == ']') {
                    return self.transitionClose(b);
                }
                return false;
            },

            .inside_literal => {
                if (self.lit_idx < self.lit_len and b == self.lit_expected[self.lit_idx]) {
                    self.lit_idx += 1;
                    if (self.lit_idx == self.lit_len) {
                        self.state = .expect_comma_or_close;
                    }
                    return true;
                }
                return false;
            },

            .expect_comma_or_close => {
                if (b == ' ' or b == '\t' or b == '\n' or b == '\r') return true;
                if (b == ',') {
                    return self.transitionComma();
                }
                if (b == '}' or b == ']') {
                    return self.transitionClose(b);
                }
                return false;
            },

            .finished => {
                // Only trailing whitespace allowed after finished JSON
                return (b == ' ' or b == '\t' or b == '\n' or b == '\r');
            },
        }
    }

    fn transitionComma(self: *JsonValidator) bool {
        if (self.depth == 0) return false;
        const cur = self.stack[self.depth - 1];
        if (cur == .object) {
            self.state = .obj_expect_key_or_close;
            return true;
        } else {
            self.state = .expect_value;
            return true;
        }
    }

    fn transitionClose(self: *JsonValidator, b: u8) bool {
        if (self.depth == 0) return false;
        const cur = self.stack[self.depth - 1];
        if (b == '}' and cur == .object) {
            self.depth -= 1;
            self.state = if (self.depth == 0) .finished else .expect_comma_or_close;
            return true;
        }
        if (b == ']' and cur == .array) {
            self.depth -= 1;
            self.state = if (self.depth == 0) .finished else .expect_comma_or_close;
            return true;
        }
        return false;
    }

    /// Tests if candidate token piece can be consumed by the validator.
    /// Returns the updated validator if valid, or null if invalid.
    pub fn tryConsume(self: JsonValidator, piece: []const u8) ?JsonValidator {
        var copy = self;
        for (piece) |b| {
            if (!copy.stepByte(b)) return null;
        }
        return copy;
    }

    /// Consumes piece in place. Returns true if valid.
    pub fn consume(self: *JsonValidator, piece: []const u8) bool {
        for (piece) |b| {
            if (!self.stepByte(b)) return false;
        }
        return true;
    }

    /// Masks invalid token logits with -infinity.
    /// Fast path: when in finished state, only EOS/EOT and whitespace tokens are allowed.
    pub fn applyMask(self: *const JsonValidator, logits: []f32, tok: *const Tokenizer, eos_id: ?u32) void {
        const neg_inf = -std.math.inf(f32);
        const is_fin = self.isFinished();

        for (logits, 0..) |*l, id_usize| {
            const id: u32 = @intCast(id_usize);
            if (is_fin) {
                if (eos_id != null and id == eos_id.?) continue;
                if (tok.isEog(id)) continue;
                const piece = tok.piece(id);
                var all_ws = piece.len > 0;
                for (piece) |b| {
                    if (b != ' ' and b != '\t' and b != '\n' and b != '\r') {
                        all_ws = false;
                        break;
                    }
                }
                if (!all_ws) l.* = neg_inf;
                continue;
            }

            // Not finished yet: EOS/EOG is forbidden before JSON is properly closed
            if (tok.isEog(id) or (eos_id != null and id == eos_id.?)) {
                l.* = neg_inf;
                continue;
            }

            const piece = tok.piece(id);
            if (piece.len == 0) continue;
            if (self.tryConsume(piece) == null) {
                l.* = neg_inf;
            }
        }
    }
};

test "JsonValidator valid and invalid sequences" {
    var jv = JsonValidator.initObjectOnly();
    try std.testing.expect(jv.consume("{\n  \"name\": \"zinfer\",\n  \"fast\": true,\n  \"version\": 2\n}"));
    try std.testing.expect(jv.isFinished());

    // Invalid starting with number for object only
    var jv_bad = JsonValidator.initObjectOnly();
    try std.testing.expect(!jv_bad.consume("123"));

    // Nested object and arrays
    var jv_nested = JsonValidator.initObjectOnly();
    try std.testing.expect(jv_nested.consume("{\"items\": [1, 2, {\"sub\": null}]}"));
    try std.testing.expect(jv_nested.isFinished());
}
