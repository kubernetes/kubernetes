// Package ebpf is a toolkit for working with eBPF programs.
//
// eBPF programs are small snippets of code which are executed directly
// in a VM in the Linux kernel, which makes them very fast and flexible.
// Many Linux subsystems now accept eBPF programs. This makes it possible
// to implement highly application specific logic inside the kernel,
// without having to modify the actual kernel itself.
//
// This package is designed for long-running processes which
// want to use eBPF to implement part of their application logic. It has no
// run-time dependencies outside of the library and the Linux kernel itself.
// eBPF code should be compiled ahead of time using clang, and shipped with
// your application as any other resource.
//
// Use the link subpackage to attach a loaded program to a hook in the kernel.
//
// Note that losing all references to Map and Program resources will cause
// their underlying file descriptors to be closed, potentially removing those
// objects from the kernel. Always retain a reference by e.g. deferring a
// Close() of a Collection or LoadAndAssign object until application exit.
//
// Special care needs to be taken when handling maps of type ProgramArray,
// as the kernel erases its contents when the last userspace or bpffs
// reference disappears, regardless of the map being in active use.
package ebpf
