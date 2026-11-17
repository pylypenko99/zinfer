//! Zero-allocation JSON Schema and Grammar-Constrained Logit Masking for zinfer.
//! Enforces RFC 8259-compliant JSON syntax (objects, arrays, strings, numbers, booleans, null)
//! directly during token sampling by setting logits of invalid tokens to -infinity.

const std = @import("std");
const Tokenizer = @import("tokenizer.zig").Tokenizer;

pub const ContainerType = enum {
    object,
    array,
};

pub const State = enum {
    start,                   // Waiting for '{' (or '[' if root is array)
    obj_expect_key_or_close, // Expecting '"' key or '}' (only right after '{')
    obj_expect_key_only,     // Expecting '"' key (after comma; trailing comma '}' is forbidden)
    inside_key,              // Inside key string until unescaped '"'
    obj_expect_colon,        // Expecting ':'
    expect_value,            // Expecting any value: '"', '{', '[', number, true, false, null
    arr_expect_val_or_close, // Expecting value or ']' (only right after '[')
    inside_str_value,        // Inside string value until unescaped '"'
    inside_num,              // Inside number
    inside_literal,          // Inside true, false, null
    expect_comma_or_close,   // Expecting ',' or '}' / ']'
    finished,                // Root container closed, only whitespace / EOS allowed
};

pub const EscapeState = enum {
    none,
    escaped, // Just read '\'
    hex1,    // After '\u'
    hex2,    // After '\uX'
    hex3,    // After '\uXX'
    hex4,    // After '\uXXX'
};

pub const NumState = enum {
    minus,       // Just read '-'
    zero,        // Read '0' (can be followed by '.' or 'e'/'E', or end)
    int_digits,  // Read '1'..'9' (can be followed by more digits, '.', 'e'/'E', or end)
    dot,         // Just read '.' (MUST be followed by at least one digit)
    frac_digits, // Read digit after '.' (can be followed by more digits, 'e'/'E', or end)
    exp,         // Just read 'e' or 'E' (can be followed by '+', '-', or digit)
    exp_sign,    // Just read '+' or '-' after 'e'/'E' (MUST be followed by at least one digit)
    exp_digits,  // Read digit in exponent (can be followed by more digits, or end)

    pub fn canTerminate(self: NumState) bool {
        return switch (self) {
            .zero, .int_digits, .frac_digits, .exp_digits => true,
            .minus, .dot, .exp, .exp_sign => false,
        };
    }
};

inline fn isHex(b: u8) bool {
    return (b >= '0' and b <= '9') or (b >= 'a' and b <= 'f') or (b >= 'A' and b <= 'F');
}

pub const JsonValidator = struct {
    state: State = .start,
    depth: usize = 0,
    stack: [64]ContainerType = undefined,
    escape_state: EscapeState = .none,
    num_state: NumState = .zero,
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

    fn stepString(self: *JsonValidator, b: u8, next_state: State) bool {
        switch (self.escape_state) {
            .none => {
                if (b == '\\') {
                    self.escape_state = .escaped;
                    return true;
                }
                if (b == '"') {
                    self.state = next_state;
                    return true;
                }
                // Control characters (< 0x20) in JSON strings are forbidden unless escaped
                if (b < 0x20) return false;
                return true;
            },
            .escaped => {
                // RFC 8259: valid escapes are ", \, /, b, f, n, r, t, u
                if (b == '"' or b == '\\' or b == '/' or b == 'b' or b == 'f' or b == 'n' or b == 'r' or b == 't') {
                    self.escape_state = .none;
                    return true;
                }
                if (b == 'u') {
                    self.escape_state = .hex1;
                    return true;
                }
                return false;
            },
            .hex1 => {
                if (isHex(b)) {
                    self.escape_state = .hex2;
                    return true;
                }
                return false;
            },
            .hex2 => {
                if (isHex(b)) {
                    self.escape_state = .hex3;
                    return true;
                }
                return false;
            },
            .hex3 => {
                if (isHex(b)) {
                    self.escape_state = .none;
                    return true;
                }
                return false;
            },
            .hex4 => unreachable,
        }
    }

    fn stepValueStart(self: *JsonValidator, b: u8) bool {
        if (b == '"') {
            self.escape_state = .none;
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
            self.state = .arr_expect_val_or_close;
            return true;
        }
        if (b == '-') {
            self.state = .inside_num;
            self.num_state = .minus;
            return true;
        }
        if (b == '0') {
            self.state = .inside_num;
            self.num_state = .zero;
            return true;
        }
        if (b >= '1' and b <= '9') {
            self.state = .inside_num;
            self.num_state = .int_digits;
            return true;
        }
        if (b == 't') {
            @memcpy(self.lit_expected[0..4], "true");
            self.lit_len = 4;
            self.lit_idx = 1;
            self.state = .inside_literal;
            return true;
        }
        if (b == 'f') {
            @memcpy(self.lit_expected[0..5], "false");
            self.lit_len = 5;
            self.lit_idx = 1;
            self.state = .inside_literal;
            return true;
        }
        if (b == 'n') {
            @memcpy(self.lit_expected[0..4], "null");
            self.lit_len = 4;
            self.lit_idx = 1;
            self.state = .inside_literal;
            return true;
        }
        return false;
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
                    self.state = .arr_expect_val_or_close;
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
                    self.escape_state = .none;
                    self.state = .inside_key;
                    return true;
                }
                return false;
            },

            .obj_expect_key_only => {
                if (b == ' ' or b == '\t' or b == '\n' or b == '\r') return true;
                // Trailing comma before '}' is forbidden by RFC 8259
                if (b == '"') {
                    self.escape_state = .none;
                    self.state = .inside_key;
                    return true;
                }
                return false;
            },

            .inside_key => {
                return self.stepString(b, .obj_expect_colon);
            },

            .obj_expect_colon => {
                if (b == ' ' or b == '\t' or b == '\n' or b == '\r') return true;
                if (b == ':') {
                    self.state = .expect_value;
                    return true;
                }
                return false;
            },

            .arr_expect_val_or_close => {
                if (b == ' ' or b == '\t' or b == '\n' or b == '\r') return true;
                if (b == ']') {
                    // Empty array []
                    if (self.depth == 0) return false;
                    self.depth -= 1;
                    self.state = if (self.depth == 0) .finished else .expect_comma_or_close;
                    return true;
                }
                return self.stepValueStart(b);
            },

            .expect_value => {
                if (b == ' ' or b == '\t' or b == '\n' or b == '\r') return true;
                return self.stepValueStart(b);
            },

            .inside_str_value => {
                return self.stepString(b, .expect_comma_or_close);
            },

            .inside_num => {
                switch (self.num_state) {
                    .minus => {
                        if (b == '0') {
                            self.num_state = .zero;
                            return true;
                        }
                        if (b >= '1' and b <= '9') {
                            self.num_state = .int_digits;
                            return true;
                        }
                        return false;
                    },
                    .zero => {
                        if (b == '.') {
                            self.num_state = .dot;
                            return true;
                        }
                        if (b == 'e' or b == 'E') {
                            self.num_state = .exp;
                            return true;
                        }
                        // Zero terminated by whitespace or delimiter
                        return self.stepNumDelimiter(b);
                    },
                    .int_digits => {
                        if (b >= '0' and b <= '9') return true;
                        if (b == '.') {
                            self.num_state = .dot;
                            return true;
                        }
                        if (b == 'e' or b == 'E') {
                            self.num_state = .exp;
                            return true;
                        }
                        return self.stepNumDelimiter(b);
                    },
                    .dot => {
                        if (b >= '0' and b <= '9') {
                            self.num_state = .frac_digits;
                            return true;
                        }
                        return false;
                    },
                    .frac_digits => {
                        if (b >= '0' and b <= '9') return true;
                        if (b == 'e' or b == 'E') {
                            self.num_state = .exp;
                            return true;
                        }
                        return self.stepNumDelimiter(b);
                    },
                    .exp => {
                        if (b == '+' or b == '-') {
                            self.num_state = .exp_sign;
                            return true;
                        }
                        if (b >= '0' and b <= '9') {
                            self.num_state = .exp_digits;
                            return true;
                        }
                        return false;
                    },
                    .exp_sign => {
                        if (b >= '0' and b <= '9') {
                            self.num_state = .exp_digits;
                            return true;
                        }
                        return false;
                    },
                    .exp_digits => {
                        if (b >= '0' and b <= '9') return true;
                        return self.stepNumDelimiter(b);
                    },
                }
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

    fn stepNumDelimiter(self: *JsonValidator, b: u8) bool {
        if (!self.num_state.canTerminate()) return false;
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
    }

    fn transitionComma(self: *JsonValidator) bool {
        if (self.depth == 0) return false;
        const cur = self.stack[self.depth - 1];
        if (cur == .object) {
            self.state = .obj_expect_key_only; // Trailing comma forbidden
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

    // Finding 8 checks:
    // 1. Incomplete number {"x":-}
    var jv_minus = JsonValidator.initObjectOnly();
    try std.testing.expect(!jv_minus.consume("{\"x\":-}"));

    // 2. Trailing comma {"x":1,}
    var jv_trail = JsonValidator.initObjectOnly();
    try std.testing.expect(!jv_trail.consume("{\"x\":1,}"));

    // 3. Illegal escape {"x":"\q"}
    var jv_esc = JsonValidator.initObjectOnly();
    try std.testing.expect(!jv_esc.consume("{\"x\":\"\\q\"}"));

    // 4. Valid escapes and unicode
    var jv_valid_esc = JsonValidator.initObjectOnly();
    try std.testing.expect(jv_valid_esc.consume("{\"text\": \"hello\\nworld\\u12af\\\"\"}"));
    try std.testing.expect(jv_valid_esc.isFinished());
}
