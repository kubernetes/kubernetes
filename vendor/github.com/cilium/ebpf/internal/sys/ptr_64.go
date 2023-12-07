//go:build !386 && !amd64p32 && !arm && !mipsle && !mips64p32le && !armbe && !mips && !mips64p32
// +build !386,!amd64p32,!arm,!mipsle,!mips64p32le,!armbe,!mips,!mips64p32

package sys

import (
	"unsafe"
)

// Pointer wraps an unsafe.Pointer to be 64bit to
// conform to the syscall specification.
type Pointer struct {
	ptr unsafe.Pointer
}
