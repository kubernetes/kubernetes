package sys

import (
	"unsafe"

	"github.com/cilium/ebpf/internal/unix"
)

// NewPointer creates a 64-bit pointer from an unsafe Pointer.
func NewPointer(ptr unsafe.Pointer) Pointer {
	return Pointer{ptr: ptr}
}

// NewSlicePointer creates a 64-bit pointer from a byte slice.
func NewSlicePointer(buf []byte) Pointer {
	if len(buf) == 0 {
		return Pointer{}
	}

	return Pointer{ptr: unsafe.Pointer(&buf[0])}
}

// NewSlicePointer creates a 64-bit pointer from a byte slice.
//
// Useful to assign both the pointer and the length in one go.
func NewSlicePointerLen(buf []byte) (Pointer, uint32) {
	return NewSlicePointer(buf), uint32(len(buf))
}

// NewStringPointer creates a 64-bit pointer from a string.
func NewStringPointer(str string) Pointer {
	p, err := unix.BytePtrFromString(str)
	if err != nil {
		return Pointer{}
	}

	return Pointer{ptr: unsafe.Pointer(p)}
}

// NewStringSlicePointer allocates an array of Pointers to each string in the
// given slice of strings and returns a 64-bit pointer to the start of the
// resulting array.
//
// Use this function to pass arrays of strings as syscall arguments.
func NewStringSlicePointer(strings []string) Pointer {
	sp := make([]Pointer, 0, len(strings))
	for _, s := range strings {
		sp = append(sp, NewStringPointer(s))
	}

	return Pointer{ptr: unsafe.Pointer(&sp[0])}
}
