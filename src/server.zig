const std = @import("std");
const Io = std.Io;
const model_mod = @import("model.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const sampler_mod = @import("sampler.zig");

pub const ServerConfig = struct {
    port: u16 = 8080,
    model_id: []const u8,
    draft_k: usize = 0,
    no_think: bool = false,
};

pub fn run(
    gpa: std.mem.Allocator,
    io: Io,
    st: *model_mod.State,
    tok: *const Tokenizer,
    smp: *sampler_mod.Sampler,
    cfg: ServerConfig,
) !void {
    const addr = try Io.net.IpAddress.parseIp4("127.0.0.1", cfg.port);
    var server = try addr.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    std.debug.print("zinfer server listening on http://127.0.0.1:{d}\n", .{cfg.port});
    std.debug.print("Model ID: {s} (ctx={d})\n", .{ cfg.model_id, st.ctx });

    while (true) {
        var stream = server.accept(io) catch |err| {
            std.debug.print("accept error: {s}\n", .{@errorName(err)});
            continue;
        };
        defer stream.close(io);

        handleConnection(gpa, io, &stream, st, tok, smp, cfg) catch |err| {
            if (err != error.EndOfStream and err != error.ConnectionAborted and err != error.WriteFailed) {
                std.debug.print("connection error: {s}\n", .{@errorName(err)});
            }
        };
    }
}

fn handleConnection(
    gpa: std.mem.Allocator,
    io: Io,
    stream: *Io.net.Stream,
    st: *model_mod.State,
    tok: *const Tokenizer,
    smp: *sampler_mod.Sampler,
    cfg: ServerConfig,
) !void {
    // Read request: read until headers end, parse Content-Length, read full body
    var req_list: std.ArrayList(u8) = .empty;
    defer req_list.deinit(gpa);

    var temp_buf: [16 * 1024]u8 = undefined;
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
                if (std.mem.indexOf(u8, headers, "Content-Length: ")) |cl_idx| {
                    const cl_str = headers[cl_idx + 16 ..];
                    const end_line = std.mem.indexOfAny(u8, cl_str, "\r\n") orelse cl_str.len;
                    content_len = std.fmt.parseInt(usize, std.mem.trim(u8, cl_str[0..end_line], " "), 10) catch null;
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
    const space1 = std.mem.indexOfScalar(u8, req, ' ') orelse return;
    const method = req[0..space1];
    const rest = req[space1 + 1 ..];
    const space2 = std.mem.indexOfScalar(u8, rest, ' ') orelse return;
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
        const json_fmt =
            \\{{"object":"list","data":[{{"id":"{s}","object":"model","created":1700000000,"owned_by":"zinfer","meta":{{"n_ctx":{d},"n_ctx_train":{d}}}}}]}}
        ;
        var body_buf: [1024]u8 = undefined;
        const body = try std.fmt.bufPrint(&body_buf, json_fmt, .{ cfg.model_id, st.ctx, st.model.cfg.ctx_train });

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
        const body = try std.fmt.bufPrint(&body_buf, json_fmt, .{st.ctx});

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
        const bs = body_start orelse return;
        const body = req[bs..];
        const is_responses_api = std.mem.indexOf(u8, path, "responses") != null;

        try handleChatCompletion(gpa, &writer.interface, body, st, tok, smp, cfg, is_responses_api);
        return;
    }

    // Unknown endpoint
    try writer.interface.writeAll("HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 0\r\n\r\n");
    try writer.interface.flush();
}

const Template = enum { llama3, chatml, lfm, plain };

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
    switch (tpl) {
        .chatml => if (no_think) try buf.appendSlice(gpa, "<think>\n\n</think>\n\n"),
        .lfm => try buf.appendSlice(gpa, if (no_think) "<think></think>" else "<think>"),
        else => {},
    }
}

fn writeJsonEscaped(w: *Io.Writer, str: []const u8) !void {
    for (str) |c| {
        switch (c) {
            '\\' => try w.writeAll("\\\\"),
            '"' => try w.writeAll("\\\""),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            0x00...0x08, 0x0b, 0x0c, 0x0e...0x1f => {
                try w.print("\\u{x:0>4}", .{c});
            },
            else => try w.writeByte(c),
        }
    }
}

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
    st: *model_mod.State,
    tok: *const Tokenizer,
    smp: *sampler_mod.Sampler,
    cfg: ServerConfig,
    is_responses_api: bool,
) !void {
    var parsed = std.json.parseFromSlice(std.json.Value, gpa, body, .{}) catch |err| {
        std.debug.print("JSON parse error: {s}\n", .{@errorName(err)});
        try w.writeAll("HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n");
        try w.flush();
        return;
    };
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return;

    var stream_mode = true;
    if (root.object.get("stream")) |sv| {
        if (sv == .bool) stream_mode = sv.bool;
    }

    var max_tokens: usize = 1024;
    if (root.object.get("max_tokens")) |mt| {
        if (mt == .integer and mt.integer > 0) max_tokens = @intCast(mt.integer);
    }

    // Get messages (or "input" for Responses API)
    const msgs_val = root.object.get("messages") orelse root.object.get("input") orelse return;
    if (msgs_val != .array) return;

    var prompt_text: std.ArrayList(u8) = .empty;
    defer prompt_text.deinit(gpa);

    const tpl = detectTemplate(tok);
    const has_think = tok.tokenId("<think>") != null;

    if (tpl == .lfm) try prompt_text.appendSlice(gpa, "<|startoftext|>");

    var text_buf: std.ArrayList(u8) = .empty;
    defer text_buf.deinit(gpa);

    for (msgs_val.array.items) |item| {
        if (item != .object) continue;
        const role_val = item.object.get("role") orelse continue;
        const content_val = item.object.get("content") orelse continue;
        if (role_val != .string) continue;

        text_buf.clearRetainingCapacity();
        try extractTextContent(gpa, content_val, &text_buf);

        try appendTurn(gpa, &prompt_text, tpl, role_val.string, text_buf.items);
    }
    try appendAssistantHeader(gpa, &prompt_text, tpl, has_think, cfg.no_think);

    // Tokenize
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    try tok.encode(gpa, prompt_text.items, true, true, &tokens);

    if (tokens.items.len == 0) return;

    st.reset();

    const prefill_logits = try st.forward(tokens.items);
    var cur_tok = smp.sample(prefill_logits);

    if (stream_mode) {
        // SSE Header
        try w.writeAll(
            "HTTP/1.1 200 OK\r\n" ++
                "Content-Type: text/event-stream; charset=utf-8\r\n" ++
                "Cache-Control: no-cache\r\n" ++
                "Connection: close\r\n" ++
                "Access-Control-Allow-Origin: *\r\n\r\n",
        );
        try w.flush();

        if (is_responses_api) {
            // Responses API streaming events
            try w.writeAll("data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n");
            try w.writeAll("data: {\"type\":\"response.content_part.added\",\"part\":{\"type\":\"text\",\"text\":\"\"}}\n\n");
            try w.flush();

            var n_gen: usize = 0;
            while (n_gen < max_tokens and st.pos < st.ctx) {
                if (tok.isEog(cur_tok)) break;

                const piece = tok.piece(cur_tok);
                try w.writeAll("data: {\"type\":\"response.output_text.delta\",\"delta\":\"");
                try writeJsonEscaped(w, piece);
                try w.writeAll("\"}\n\n");
                try w.flush();

                smp.accept(cur_tok);
                n_gen += 1;

                const logits = try st.forward(&.{cur_tok});
                cur_tok = smp.sample(logits);
            }

            try w.writeAll("data: {\"type\":\"response.completed\"}\n\n");
            try w.writeAll("data: [DONE]\n\n");
            try w.flush();
        } else {
            // Chat completions API streaming chunks
            try w.writeAll("data: {\"id\":\"chatcmpl-zinfer\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n");
            try w.flush();

            var n_gen: usize = 0;
            while (n_gen < max_tokens and st.pos < st.ctx) {
                if (tok.isEog(cur_tok)) break;

                const piece = tok.piece(cur_tok);
                try w.writeAll("data: {\"id\":\"chatcmpl-zinfer\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"");
                try writeJsonEscaped(w, piece);
                try w.writeAll("\"},\"finish_reason\":null}]}\n\n");
                try w.flush();

                smp.accept(cur_tok);
                n_gen += 1;

                const logits = try st.forward(&.{cur_tok});
                cur_tok = smp.sample(logits);
            }

            try w.writeAll("data: {\"id\":\"chatcmpl-zinfer\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n");
            try w.writeAll("data: [DONE]\n\n");
            try w.flush();
        }
    } else {
        var resp_text: std.ArrayList(u8) = .empty;
        defer resp_text.deinit(gpa);

        var n_gen: usize = 0;
        while (n_gen < max_tokens and st.pos < st.ctx) {
            if (tok.isEog(cur_tok)) break;
            try resp_text.appendSlice(gpa, tok.piece(cur_tok));
            smp.accept(cur_tok);
            n_gen += 1;
            const logits = try st.forward(&.{cur_tok});
            cur_tok = smp.sample(logits);
        }

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
            "\"}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d}}}}}",
            .{ tokens.items.len, n_gen, tokens.items.len + n_gen },
        );
        try w.flush();
    }
}
