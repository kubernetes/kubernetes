// Code generated mksyscall_windows.exe DO NOT EDIT

package hcs

import (
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

var _ unsafe.Pointer

// Do the interface allocations only once for common
// Errno values.
const (
	errnoERROR_IO_PENDING = 997
)

var (
	errERROR_IO_PENDING error = syscall.Errno(errnoERROR_IO_PENDING)
)

// errnoErr returns common boxed Errno values, to prevent
// allocations at runtime.
func errnoErr(e syscall.Errno) error {
	switch e {
	case 0:
		return nil
	case errnoERROR_IO_PENDING:
		return errERROR_IO_PENDING
	}
	// TODO: add more here, after collecting data on the common
	// error values see on Windows. (perhaps when running
	// all.bat?)
	return e
}

var (
	modcomputestorage = windows.NewLazySystemDLL("computestorage.dll")

	procHcsFormatWritableLayerVhd = modcomputestorage.NewProc("HcsFormatWritableLayerVhd")
)

func hcsFormatWritableLayerVhd(handle uintptr) (hr error) {
	r0, _, _ := syscall.Syscall(procHcsFormatWritableLayerVhd.Addr(), 1, uintptr(handle), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}
