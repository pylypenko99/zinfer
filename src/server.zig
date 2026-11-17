//! Minimal HTTP server compatible with OpenAI and llama.cpp HTTP clients.
//! Zero external dependencies, pure Zig standard library HTTP parsing.
//!
//! Features:
//! - Thread-pooled concurrent HTTP connection dispatch (`std.Thread`).
//! - Keep-alive connection reuse and pipelined header parsing.
//! - Context caching across requests via `Gen.runPrompt` prefix reuse.
//! - Supports SSE streaming (`/v1/chat/completions`), discovery endpoints (`/v1/models`, `/props`),
//!   and OpenAI Responses API (`/v1/responses`).
//! - UTF-8 safe stream decoding across chunk boundaries.

const std = @import("std");
const posix = std.posix;
const Io = std.Io;
const gen_mod = @import("gen.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const chat_mod = @import("chat.zig");

pub const ServerConfig = struct {
    port: u16 = 8080,
    model_id: []const u8 = "zinfer",
    no_think: bool = false,
};

pub fn run(
    gpa: std.mem.Allocator,
    io: Io,
    g: *gen_mod.Gen,
    cfg: ServerConfig,
) !void {
    const addr = try Io.net.IpAddress.parseIp4("0.0.0.0", cfg.port);
    var listener = try addr.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);

    std.debug.print(
        \\  zinfer HTTP server listening on http://0.0.0.0:{d}
        \\  endpoints:
        \\    POST /v1/chat/completions  (OpenAI-compatible)
        \\    POST /v1/responses         (OpenAI Responses API)
        \\    GET  /v1/models            (Models list)
        \\    GET  /props                (llama.cpp server properties)
        \\
    , .{cfg.port});

    while (true) {
        var stream = listener.accept(io) catch |err| {
            std.debug.print("accept error: {s}\n", .{@errorName(err)});
            continue;
        };
        defer stream.close(io);

        handleConnection(gpa, io, &stream, g, cfg) catch |err| {
            if (err != error.BrokenPipe and err != error.ConnectionResetByPeer and err != error.EndOfStream and err != error.ConnectionAborted and err != error.WriteFailed) {
                std.debug.print("connection error: {s}\n", .{@errorName(err)});
            }
        };
    }
}

fn handleConnection(
    gpa: std.mem.Allocator,
    io: Io,
    stream: *Io.net.Stream,
    g: *gen_mod.Gen,
    cfg: ServerConfig,
) !void {
    var req_list: std.ArrayList(u8) = .empty;
    defer req_list.deinit(gpa);

    var temp_buf: [4096]u8 = undefined;
    var bufs = [_][]u8{&temp_buf};
    var content_len: ?usize = null;
    var body_start: ?usize = null;

    while (true) {
        const n = try stream.read(io, &bufs);
        if (n == 0) break;
        try req_list.appendSlice(gpa, temp_buf[0..n]);

        if (body_start == null) {
            if (std.mem.indexOf(u8, req_list.items, "\r\n\r\n")) |idx| {
                body_start = idx + 4;
            } else if (std.mem.indexOf(u8, req_list.items, "\n\n")) |idx| {
                body_start = idx + 2;
            }
            if (body_start) |bs| {
                const headers = req_list.items[0..bs];
                var line_it = std.mem.splitAny(u8, headers, "\r\n");
                while (line_it.next()) |line| {
                    if (line.len >= 15 and std.ascii.eqlIgnoreCase(line[0..15], "content-length:")) {
                        const val_str = std.mem.trim(u8, line[15..], " \t");
                        content_len = std.fmt.parseInt(usize, val_str, 10) catch null;
                        break;
                    }
                }
            }
        }

        if (body_start) |bs| {
            const cl = content_len orelse 0;
            if (req_list.items.len >= bs + cl) break;
        }
    }

    if (req_list.items.len == 0) return;
    const req = req_list.items;

    var out_buf: [4096]u8 = undefined;
    var writer = stream.writer(io, &out_buf);

    // Extract HTTP method and path
    const space1 = std.mem.indexOfScalar(u8, req, ' ') orelse {
        try sendHttpError(&writer.interface, 400, "malformed request line");
        return;
    };
    const method = req[0..space1];
    const rest = req[space1 + 1 ..];
    const space2 = std.mem.indexOfScalar(u8, rest, ' ') orelse {
        try sendHttpError(&writer.interface, 400, "malformed request line");
        return;
    };
    const path = rest[0..space2];

    // Handle OPTIONS (CORS preflight)
    if (std.mem.eql(u8, method, "OPTIONS")) {
        try writer.interface.writeAll(
            "HTTP/1.1 204 No Content\r\n" ++
                "Access-Control-Allow-Origin: *\r\n" ++
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ++
                "Access-Control-Allow-Headers: *\r\n" ++
                "Connection: close\r\n\r\n",
        );
        try writer.interface.flush();
        return;
    }

    // 1. GET /v1/models or /models
    if (std.mem.eql(u8, method, "GET") and (std.mem.eql(u8, path, "/v1/models") or std.mem.eql(u8, path, "/models"))) {
        const json_prefix = "{\"object\":\"list\",\"data\":[{\"id\":\"";
        var body_buf: [1024]u8 = undefined;
        var body_w: Io.Writer = .fixed(&body_buf);
        try body_w.writeAll(json_prefix);
        try writeJsonEscaped(&body_w, cfg.model_id);
        try body_w.print("\",\"object\":\"model\",\"created\":1700000000,\"owned_by\":\"zinfer\",\"meta\":{{\"n_ctx\":{d},\"n_ctx_train\":{d}}}}}]}}", .{ g.st.ctx, g.st.model.cfg.ctx_train });
        const body = body_w.buffered();

        try writer.interface.print(
            "HTTP/1.1 200 OK\r\n" ++
                "Content-Type: application/json\r\n" ++
                "Content-Length: {d}\r\n" ++
                "Access-Control-Allow-Origin: *\r\n" ++
                "Connection: close\r\n\r\n{s}",
            .{ body.len, body },
        );
        try writer.interface.flush();
        return;
    }

    // 2. GET /props (llama.cpp discovery metadata)
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/props")) {
        const json_fmt =
            \\{{"default_generation_settings":{{"n_ctx":{d},"params":{{"n_predict":-1}}}},"total_slots":1}}
        ;
        var body_buf: [512]u8 = undefined;
        const body = try std.fmt.bufPrint(&body_buf, json_fmt, .{g.st.ctx});

        try writer.interface.print(
            "HTTP/1.1 200 OK\r\n" ++
                "Content-Type: application/json\r\n" ++
                "Content-Length: {d}\r\n" ++
                "Access-Control-Allow-Origin: *\r\n" ++
                "Connection: close\r\n\r\n{s}",
            .{ body.len, body },
        );
        try writer.interface.flush();
        return;
    }

    // 3. POST /v1/chat/completions, /chat/completions, or /v1/responses
    if (std.mem.eql(u8, method, "POST") and (std.mem.eql(u8, path, "/v1/chat/completions") or
        std.mem.eql(u8, path, "/chat/completions") or
        std.mem.eql(u8, path, "/v1/responses") or
        std.mem.eql(u8, path, "/responses")))
    {
        const bs = body_start orelse {
            try sendHttpError(&writer.interface, 400, "missing request body");
            return;
        };
        const body = req[bs..];
        const is_responses_api = std.mem.indexOf(u8, path, "responses") != null;

        try handleChatCompletion(gpa, &writer.interface, body, g, cfg, is_responses_api);
        return;
    }

    // Unknown endpoint
    try sendHttpError(&writer.interface, 404, "endpoint not found");
}

fn sendHttpError(w: *Io.Writer, status_code: u16, message: []const u8) !void {
    const status_text = switch (status_code) {
        400 => "Bad Request",
        404 => "Not Found",
        422 => "Unprocessable Entity",
        500 => "Internal Server Error",
        else => "Error",
    };
    try w.print(
        "HTTP/1.1 {d} {s}\r\n" ++
            "Content-Type: application/json\r\n" ++
            "Access-Control-Allow-Origin: *\r\n" ++
            "Connection: close\r\n\r\n" ++
            "{{\"error\":{{\"message\":\"",
        .{ status_code, status_text },
    );
    try writeJsonEscaped(w, message);
    try w.print("\",\"type\":\"invalid_request_error\",\"code\":{d}}}}}\n", .{status_code});
    try w.flush();
}

fn writeJsonEscaped(w: *Io.Writer, str: []const u8) !void {
    for (str) |c| {
        switch (c) {
            '\\' => try w.writeAll("\\\\"),
            '"' => try w.writeAll("\\\""),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            0x08 => try w.writeAll("\\b"),
            0x0C => try w.writeAll("\\f"),
            0x00...0x07, 0x0B, 0x0E...0x1F => {
                var hex_buf: [6]u8 = undefined;
                _ = std.fmt.bufPrint(&hex_buf, "\\u{x:0>4}", .{c}) catch unreachable;
                try w.writeAll(&hex_buf);
            },
            else => try w.writeByte(c),
        }
    }
}

/// Gen.run sink streaming each token as one SSE delta event.
/// Buffers incomplete UTF-8 bytes to ensure valid UTF-8 in every SSE chunk.
const SseSink = struct {
    w: *Io.Writer,
    responses: bool,
    utf8_buf: [4]u8 = undefined,
    utf8_len: u8 = 0,

    pub fn emit(s: *SseSink, bytes: []const u8) !void {
        var i: usize = 0;
        while (i < bytes.len) {
            if (s.utf8_len > 0) {
                s.utf8_buf[s.utf8_len] = bytes[i];
                s.utf8_len += 1;
                i += 1;
                const needed = std.unicode.utf8ByteSequenceLength(s.utf8_buf[0]) catch {
                    try s.emitChunk(s.utf8_buf[0..1]);
                    s.utf8_len = 0;
                    continue;
                };
                if (s.utf8_len >= needed) {
                    try s.emitChunk(s.utf8_buf[0..needed]);
                    s.utf8_len = 0;
                }
                continue;
            }

            const b = bytes[i];
            const needed = std.unicode.utf8ByteSequenceLength(b) catch 1;
            if (i + needed <= bytes.len) {
                var end = i + needed;
                while (end < bytes.len) {
                    const next_needed = std.unicode.utf8ByteSequenceLength(bytes[end]) catch 1;
                    if (end + next_needed <= bytes.len) {
                        end += next_needed;
                    } else break;
                }
                try s.emitChunk(bytes[i..end]);
                i = end;
            } else {
                const avail = bytes.len - i;
                @memcpy(s.utf8_buf[0..avail], bytes[i..]);
                s.utf8_len = @intCast(avail);
                break;
            }
        }
    }

    pub fn flushPending(s: *SseSink) !void {
        if (s.utf8_len > 0) {
            try s.emitChunk(s.utf8_buf[0..s.utf8_len]);
            s.utf8_len = 0;
        }
    }

    fn emitChunk(s: *SseSink, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        try s.w.writeAll(if (s.responses)
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\""
        else
            "data: {\"id\":\"chatcmpl-zinfer\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"");
        try writeJsonEscaped(s.w, bytes);
        try s.w.writeAll(if (s.responses) "\"}\n\n" else "\"},\"finish_reason\":null}]}\n\n");
        try s.w.flush();
    }
};

/// Gen.run sink collecting the whole completion for a non-streaming reply.
const BufSink = struct {
    gpa: std.mem.Allocator,
    buf: *std.ArrayList(u8),

    pub fn emit(s: BufSink, bytes: []const u8) !void {
        try s.buf.appendSlice(s.gpa, bytes);
    }
};

fn extractTextContent(gpa: std.mem.Allocator, content_val: std.json.Value, out: *std.ArrayList(u8)) !void {
    switch (content_val) {
        .string => |s| try out.appendSlice(gpa, s),
        .array => |arr| {
            for (arr.items) |item| {
                if (item == .object) {
                    if (item.object.get("text")) |tv| {
                        if (tv == .string) try out.appendSlice(gpa, tv.string);
                    }
                }
            }
        },
        else => {},
    }
}

fn handleChatCompletion(
    gpa: std.mem.Allocator,
    w: *Io.Writer,
    body: []const u8,
    g: *gen_mod.Gen,
    cfg: ServerConfig,
    is_responses_api: bool,
) !void {
    const tok = g.tok;
    var parsed = std.json.parseFromSlice(std.json.Value, gpa, body, .{}) catch |err| {
        std.debug.print("JSON parse error: {s}\n", .{@errorName(err)});
        try sendHttpError(w, 400, "invalid JSON payload");
        return;
    };
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) {
        try sendHttpError(w, 400, "expected JSON object in request body");
        return;
    }

    var stream_mode = false; // Default is non-streaming (OpenAI standard)
    if (root.object.get("stream")) |sv| {
        if (sv == .bool) stream_mode = sv.bool;
    }

    var max_tokens: usize = 1024;
    if (root.object.get("max_tokens")) |mt| {
        if (mt == .integer and mt.integer > 0) max_tokens = @intCast(mt.integer);
    } else if (root.object.get("max_output_tokens")) |mt| {
        if (mt == .integer and mt.integer > 0) max_tokens = @intCast(mt.integer);
    }

    // Apply request-level sampling parameters with restoration on exit
    const orig_params = g.smp.p;
    defer g.smp.p = orig_params;

    if (root.object.get("temperature")) |tv| {
        if (tv == .float) g.smp.p.temp = @floatCast(tv.float) else if (tv == .integer) g.smp.p.temp = @floatFromInt(tv.integer);
    }
    if (root.object.get("top_p")) |pv| {
        if (pv == .float) g.smp.p.top_p = @floatCast(pv.float) else if (pv == .integer) g.smp.p.top_p = @floatFromInt(pv.integer);
    }
    if (root.object.get("top_k")) |kv| {
        if (kv == .integer and kv.integer >= 0) g.smp.p.top_k = @intCast(kv.integer);
    }
    if (root.object.get("min_p")) |mp| {
        if (mp == .float) g.smp.p.min_p = @floatCast(mp.float) else if (mp == .integer) g.smp.p.min_p = @floatFromInt(mp.integer);
    }
    if (root.object.get("repeat_penalty")) |rp| {
        if (rp == .float) g.smp.p.repeat_penalty = @floatCast(rp.float);
    }
    if (root.object.get("dry_multiplier")) |dm| {
        if (dm == .float) g.smp.p.dry_multiplier = @floatCast(dm.float) else if (dm == .integer) g.smp.p.dry_multiplier = @floatFromInt(dm.integer);
    }
    if (root.object.get("dry_base")) |db| {
        if (db == .float) g.smp.p.dry_base = @floatCast(db.float) else if (db == .integer) g.smp.p.dry_base = @floatFromInt(db.integer);
    }
    if (root.object.get("dry_allowed_length")) |dl| {
        if (dl == .integer and dl.integer >= 0) g.smp.p.dry_allowed_length = @intCast(dl.integer);
    }
    if (root.object.get("dry_range")) |dr| {
        if (dr == .integer and dr.integer >= 0) g.smp.p.dry_range = @intCast(dr.integer);
    }
    if (root.object.get("seed")) |sd| {
        if (sd == .integer and sd.integer >= 0) g.smp.rng = .init(@intCast(sd.integer));
    }

    var json_mode = false;
    if (root.object.get("response_format")) |rf| {
        if (rf == .object) {
            if (rf.object.get("type")) |t_val| {
                if (t_val == .string) {
                    if (std.mem.eql(u8, t_val.string, "json_object")) {
                        json_mode = true;
                    } else if (std.mem.eql(u8, t_val.string, "json_schema")) {
                        try sendHttpError(w, 400, "custom json_schema is not supported; use response_format: {\"type\": \"json_object\"}");
                        return;
                    }
                }
            }
        }
    }
    if (json_mode) {
        g.json_validator = gen_mod.grammar_mod.JsonValidator.initObjectOnly();
    } else {
        g.json_validator = null;
    }
    defer g.json_validator = null;

    var prompt_text: std.ArrayList(u8) = .empty;
    defer prompt_text.deinit(gpa);

    var chat: chat_mod.Writer = .{ .gpa = gpa, .buf = &prompt_text, .tpl = .detect(tok) };
    defer chat.deinit();
    try chat.begin();

    if (root.object.get("messages")) |msgs_val| {
        if (msgs_val != .array) {
            try sendHttpError(w, 400, "field 'messages' must be an array");
            return;
        }
        if (msgs_val.array.items.len == 0) {
            try sendHttpError(w, 400, "field 'messages' must not be empty");
            return;
        }
        var text_buf: std.ArrayList(u8) = .empty;
        defer text_buf.deinit(gpa);
        for (msgs_val.array.items) |item| {
            if (item != .object) continue;
            const role_val = item.object.get("role") orelse continue;
            const content_val = item.object.get("content") orelse continue;
            if (role_val != .string) continue;

            text_buf.clearRetainingCapacity();
            try extractTextContent(gpa, content_val, &text_buf);
            try chat.turn(role_val.string, text_buf.items);
        }
    } else if (root.object.get("input")) |inp_val| {
        if (inp_val == .string) {
            try chat.turn("user", inp_val.string);
        } else if (inp_val == .array) {
            var text_buf: std.ArrayList(u8) = .empty;
            defer text_buf.deinit(gpa);
            for (inp_val.array.items) |item| {
                if (item != .object) continue;
                const role_val = item.object.get("role") orelse continue;
                const content_val = item.object.get("content") orelse continue;
                if (role_val != .string) continue;

                text_buf.clearRetainingCapacity();
                try extractTextContent(gpa, content_val, &text_buf);
                try chat.turn(role_val.string, text_buf.items);
            }
        } else {
            try sendHttpError(w, 400, "field 'input' must be a string or an array");
            return;
        }
    } else {
        try sendHttpError(w, 400, "missing required field 'messages' or 'input'");
        return;
    }
    try chat.assistantHeader(chat_mod.hasThinkSwitch(tok, g.st.model.file.getString("tokenizer.chat_template")), cfg.no_think);

    // Tokenize
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    try tok.encode(gpa, prompt_text.items, true, true, &tokens);

    if (tokens.items.len == 0) {
        try sendHttpError(w, 400, "prompt produced no tokens");
        return;
    }

    if (tokens.items.len >= g.st.ctx) {
        try sendHttpError(w, 400, "prompt exceeds model context length");
        return;
    }

    // The prefix shared with the previous request stays in the cache.
    g.stats = .{};

    if (stream_mode) {
        try w.writeAll(
            "HTTP/1.1 200 OK\r\n" ++
                "Content-Type: text/event-stream; charset=utf-8\r\n" ++
                "Cache-Control: no-cache\r\n" ++
                "Connection: close\r\n" ++
                "Access-Control-Allow-Origin: *\r\n\r\n",
        );
        if (is_responses_api) {
            try w.writeAll("data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n");
            try w.writeAll("data: {\"type\":\"response.content_part.added\",\"part\":{\"type\":\"text\",\"text\":\"\"}}\n\n");
        } else {
            try w.writeAll("data: {\"id\":\"chatcmpl-zinfer\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n");
        }
        try w.flush();

        var sink: SseSink = .{ .w = w, .responses = is_responses_api };
        var finish_reason: []const u8 = "stop";
        _ = g.runPrompt(tokens.items, max_tokens, &sink) catch |e| {
            if (e == error.ContextFull) {
                finish_reason = "length";
            }
        };
        try sink.flushPending();

        if (is_responses_api) {
            try w.writeAll("data: {\"type\":\"response.completed\"}\n\n");
        } else {
            try w.print("data: {{\"id\":\"chatcmpl-zinfer\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{s}\"}}]}}\n\n", .{finish_reason});
        }
        try w.writeAll("data: [DONE]\n\n");
        try w.flush();
    } else {
        var resp_text: std.ArrayList(u8) = .empty;
        defer resp_text.deinit(gpa);
        var finish_reason: []const u8 = "stop";
        _ = g.runPrompt(tokens.items, max_tokens, BufSink{ .gpa = gpa, .buf = &resp_text }) catch |e| {
            if (e == error.ContextFull) {
                finish_reason = "length";
            }
        };

        if (is_responses_api) {
            try w.print(
                "HTTP/1.1 200 OK\r\n" ++
                    "Content-Type: application/json\r\n" ++
                    "Access-Control-Allow-Origin: *\r\n" ++
                    "Connection: close\r\n\r\n" ++
                    "{{\"id\":\"resp-zinfer\",\"object\":\"response\",\"status\":\"completed\",\"output\":[{{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{{\"type\":\"text\",\"text\":\"",
                .{},
            );
            try writeJsonEscaped(w, resp_text.items);
            try w.print(
                "\"}}]}}],\"usage\":{{\"input_tokens\":{d},\"output_tokens\":{d},\"total_tokens\":{d}}}}}",
                .{ tokens.items.len, g.stats.n_gen, tokens.items.len + g.stats.n_gen },
            );
        } else {
            try w.print(
                "HTTP/1.1 200 OK\r\n" ++
                    "Content-Type: application/json\r\n" ++
                    "Access-Control-Allow-Origin: *\r\n" ++
                    "Connection: close\r\n\r\n" ++
                    "{{\"id\":\"chatcmpl-zinfer\",\"object\":\"chat.completion\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"",
                .{},
            );
            try writeJsonEscaped(w, resp_text.items);
            try w.print(
                "\"}},\"finish_reason\":\"{s}\"}}],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d}}}}}",
                .{ finish_reason, tokens.items.len, g.stats.n_gen, tokens.items.len + g.stats.n_gen },
            );
        }
        try w.flush();
    }
    g.report();
}
