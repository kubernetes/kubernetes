// +build armbe mips mips64p32

package ebpf

import (
	"unsafe"
)

// ptr wraps an unsafe.Pointer to be 64bit to
// conform to the syscall specification.
type syscallPtr struct {
	pad uint32
	ptr unsafe.Pointer
}
