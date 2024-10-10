package sysenc

import (
	"unsafe"

	"github.com/cilium/ebpf/internal/sys"
)

type Buffer struct {
	ptr unsafe.Pointer
	// Size of the buffer. syscallPointerOnly if created from UnsafeBuffer or when using
	// zero-copy unmarshaling.
	size int
}

const syscallPointerOnly = -1

func newBuffer(buf []byte) Buffer {
	if len(buf) == 0 {
		return Buffer{}
	}
	return Buffer{unsafe.Pointer(&buf[0]), len(buf)}
}

// UnsafeBuffer constructs a Buffer for zero-copy unmarshaling.
//
// [Pointer] is the only valid method to call on such a Buffer.
// Use [SyscallBuffer] instead if possible.
func UnsafeBuffer(ptr unsafe.Pointer) Buffer {
	return Buffer{ptr, syscallPointerOnly}
}

// SyscallOutput prepares a Buffer for a syscall to write into.
//
// size is the length of the desired buffer in bytes.
// The buffer may point at the underlying memory of dst, in which case [Unmarshal]
// becomes a no-op.
//
// The contents of the buffer are undefined and may be non-zero.
func SyscallOutput(dst any, size int) Buffer {
	if dstBuf := unsafeBackingMemory(dst); len(dstBuf) == size {
		buf := newBuffer(dstBuf)
		buf.size = syscallPointerOnly
		return buf
	}

	return newBuffer(make([]byte, size))
}

// CopyTo copies the buffer into dst.
//
// Returns the number of copied bytes.
func (b Buffer) CopyTo(dst []byte) int {
	return copy(dst, b.unsafeBytes())
}

// AppendTo appends the buffer onto dst.
func (b Buffer) AppendTo(dst []byte) []byte {
	return append(dst, b.unsafeBytes()...)
}

// Pointer returns the location where a syscall should write.
func (b Buffer) Pointer() sys.Pointer {
	// NB: This deliberately ignores b.length to support zero-copy
	// marshaling / unmarshaling using unsafe.Pointer.
	return sys.NewPointer(b.ptr)
}

// Unmarshal the buffer into the provided value.
func (b Buffer) Unmarshal(data any) error {
	if b.size == syscallPointerOnly {
		return nil
	}

	return Unmarshal(data, b.unsafeBytes())
}

func (b Buffer) unsafeBytes() []byte {
	if b.size == syscallPointerOnly {
		return nil
	}
	return unsafe.Slice((*byte)(b.ptr), b.size)
}
