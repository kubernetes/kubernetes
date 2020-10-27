package internal

import "unsafe"

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

// NewStringPointer creates a 64-bit pointer from a string.
func NewStringPointer(str string) Pointer {
	if str == "" {
		return Pointer{}
	}

	// The kernel expects strings to be zero terminated
	buf := make([]byte, len(str)+1)
	copy(buf, str)

	return Pointer{ptr: unsafe.Pointer(&buf[0])}
}
