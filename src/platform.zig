//! Cross-platform OS abstractions for zinfer (Linux & Windows).
//! Provides memory mapping (mmap / VirtualAlloc) and console signal handlers
//! without external C runtime dependencies (`link_libc = false`).

const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

pub const is_windows = builtin.os.tag == .windows;

pub const win = if (is_windows) struct {
    pub const HANDLE = std.os.windows.HANDLE;
    pub extern "kernel32" fn CreateFileMappingA(
        hFile: HANDLE,
        lpAttributes: ?*anyopaque,
        flProtect: u32,
        dwMaximumSizeHigh: u32,
        dwMaximumSizeLow: u32,
        lpName: ?[*:0]const u8,
    ) callconv(.winapi) ?HANDLE;

    pub extern "kernel32" fn MapViewOfFile(
        hFileMappingObject: HANDLE,
        dwDesiredAccess: u32,
        dwFileOffsetHigh: u32,
        dwFileOffsetLow: u32,
        dwNumberOfBytesToMap: usize,
    ) callconv(.winapi) ?*anyopaque;

    pub extern "kernel32" fn UnmapViewOfFile(
        lpBaseAddress: ?*const anyopaque,
    ) callconv(.winapi) i32;

    pub extern "kernel32" fn CloseHandle(
        hObject: HANDLE,
    ) callconv(.winapi) i32;

    pub extern "kernel32" fn VirtualAlloc(
        lpAddress: ?*anyopaque,
        dwSize: usize,
        flAllocationType: u32,
        flProtect: u32,
    ) callconv(.winapi) ?*anyopaque;

    pub extern "kernel32" fn VirtualFree(
        lpAddress: ?*anyopaque,
        dwSize: usize,
        dwFreeType: u32,
    ) callconv(.winapi) i32;

    pub extern "kernel32" fn SetConsoleCtrlHandler(
        HandlerRoutine: ?*const fn (u32) callconv(.winapi) i32,
        Add: i32,
    ) callconv(.winapi) i32;
} else struct {};

pub fn mapFileReadOnly(handle: std.Io.File.Handle, size: usize) ![]align(std.heap.page_size_min) const u8 {
    if (is_windows) {
        // 0x02 = PAGE_READONLY
        const hMap = win.CreateFileMappingA(handle, null, 0x02, 0, 0, null) orelse return error.SystemResources;
        defer _ = win.CloseHandle(hMap);
        // 0x04 = FILE_MAP_READ
        const ptr = win.MapViewOfFile(hMap, 0x04, 0, 0, size) orelse return error.SystemResources;
        const byte_ptr: [*]align(std.heap.page_size_min) const u8 = @ptrCast(@alignCast(ptr));
        return byte_ptr[0..size];
    } else {
        return posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, handle, 0);
    }
}

pub fn unmapFile(slice: []align(std.heap.page_size_min) const u8) void {
    if (is_windows) {
        _ = win.UnmapViewOfFile(slice.ptr);
    } else {
        posix.munmap(slice);
    }
}

pub fn allocAnonymous(len: usize) ![]align(std.heap.page_size_min) u8 {
    if (is_windows) {
        // 0x1000 = MEM_COMMIT, 0x2000 = MEM_RESERVE, 0x04 = PAGE_READWRITE
        const ptr = win.VirtualAlloc(null, len, 0x1000 | 0x2000, 0x04) orelse return error.OutOfMemory;
        const byte_ptr: [*]align(std.heap.page_size_min) u8 = @ptrCast(@alignCast(ptr));
        return byte_ptr[0..len];
    } else {
        const region = try posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        posix.madvise(region.ptr, len, posix.MADV.HUGEPAGE) catch {};
        return region;
    }
}

pub fn freeAnonymous(slice: []align(std.heap.page_size_min) u8) void {
    if (is_windows) {
        // 0x8000 = MEM_RELEASE
        _ = win.VirtualFree(slice.ptr, 0, 0x8000);
    } else {
        posix.munmap(slice);
    }
}

pub fn setupCtrlHandler(handler: *const fn (u32) callconv(.winapi) i32) void {
    if (is_windows) {
        _ = win.SetConsoleCtrlHandler(handler, 1);
    }
}
