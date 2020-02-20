// +build !386,!amd64p32,!arm,!mipsle,!mips64p32le
// +build !armbe,!mips,!mips64p32

package ebpf

import (
	"unsafe"
)

// ptr wraps an unsafe.Pointer to be 64bit to
// conform to the syscall specification.
type syscallPtr struct {
	ptr unsafe.Pointer
}
