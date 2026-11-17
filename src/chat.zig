//! Chat prompt templates shared by the CLI and the server, picked from the
//! special tokens of the vocabulary.
const std = @import("std");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Allocator = std.mem.Allocator;

pub const Template = enum {
    llama3,
    chatml,
    /// ChatML preceded by <|startoftext|> (Liquid LFM).
    lfm,
    /// <start_of_turn>role ... <end_of_turn>: assistant turns are "model" and
    /// a system message becomes a prefix of the next turn.
    gemma,
    /// <|turn>role ... <turn|>: Gemma 4 format.
    gemma4,
    plain,

    pub fn detect(tok: *const Tokenizer) Template {
        if (tok.tokenId("<|turn>") != null and tok.tokenId("<turn|>") != null) return .gemma4;
        if (tok.tokenId("<|start_header_id|>") != null and tok.tokenId("<|eot_id|>") != null) return .llama3;
        if (tok.tokenId("<|im_start|>") != null and tok.tokenId("<|im_end|>") != null)
            return if (tok.tokenId("<|startoftext|>") != null) .lfm else .chatml;
        if (tok.tokenId("<start_of_turn>") != null and tok.tokenId("<end_of_turn>") != null) return .gemma;
        return .plain;
    }

    /// Text the template puts after a turn's end-of-turn token, which the
    /// generated end-of-turn token is fed back without.
    pub fn turnSeparator(tpl: Template) []const u8 {
        return switch (tpl) {
            .chatml, .lfm, .gemma, .gemma4 => "\n",
            .llama3, .plain => "",
        };
    }
};

/// Whether the generation prompt opens a think block: the vocabulary has a
/// <think> token and the chat template an enable_thinking switch. Instruct
/// models sharing a thinking tokenizer (LFM2.5) have the token but not the
/// switch; without a template the token alone decides.
pub fn hasThinkSwitch(tok: *const Tokenizer, template: ?[]const u8) bool {
    if (tok.tokenId("<think>") == null) return false;
    const t = template orelse return true;
    return std.mem.indexOf(u8, t, "enable_thinking") != null;
}

/// Renders a conversation into `buf` turn by turn.
pub const Writer = struct {
    gpa: Allocator,
    buf: *std.ArrayList(u8),
    tpl: Template,
    /// Gemma's system message, held until the next turn opens.
    system: std.ArrayList(u8) = .empty,

    pub fn deinit(w: *Writer) void {
        w.system.deinit(w.gpa);
    }

    /// Text that opens a conversation, before its first turn.
    pub fn begin(w: *Writer) !void {
        if (w.tpl == .lfm) try w.buf.appendSlice(w.gpa, "<|startoftext|>");
    }

    pub fn turn(w: *Writer, role: []const u8, msg: []const u8) !void {
        switch (w.tpl) {
            .llama3 => try w.buf.print(w.gpa, "<|start_header_id|>{s}<|end_header_id|>\n\n{s}<|eot_id|>", .{ role, msg }),
            .chatml, .lfm => try w.buf.print(w.gpa, "<|im_start|>{s}\n{s}<|im_end|>\n", .{ role, msg }),
            .gemma, .gemma4 => {
                if (std.mem.eql(u8, role, "system")) {
                    w.system.clearRetainingCapacity();
                    try w.system.print(w.gpa, "{s}\n\n", .{msg});
                    return;
                }
                const r = if (std.mem.eql(u8, role, "assistant")) "model" else role;
                const open_tag = if (w.tpl == .gemma4) "<|turn>" else "<start_of_turn>";
                const close_tag = if (w.tpl == .gemma4) "<turn|>" else "<end_of_turn>";
                try w.buf.print(w.gpa, "{s}{s}\n{s}{s}{s}\n", .{ open_tag, r, w.system.items, std.mem.trim(u8, msg, &std.ascii.whitespace), close_tag });
                w.system.clearRetainingCapacity();
            },
            .plain => try w.buf.print(w.gpa, "{s}: {s}\n", .{ role, msg }),
        }
    }

    /// Opens the assistant turn to generate. With a think token, `no_think`
    /// starts it with an empty think block (each template's enable_thinking).
    pub fn assistantHeader(w: *Writer, has_think: bool, no_think: bool) !void {
        try w.buf.appendSlice(w.gpa, switch (w.tpl) {
            .llama3 => "<|start_header_id|>assistant<|end_header_id|>\n\n",
            .chatml, .lfm => "<|im_start|>assistant\n",
            .gemma => "<start_of_turn>model\n",
            .gemma4 => "<|turn>model\n",
            .plain => "assistant:",
        });
        if (!has_think) return;
        switch (w.tpl) {
            .chatml => if (no_think) try w.buf.appendSlice(w.gpa, "<think>\n\n</think>\n\n"),
            .lfm => try w.buf.appendSlice(w.gpa, if (no_think) "<think></think>" else "<think>"),
            else => {},
        }
    }
};
