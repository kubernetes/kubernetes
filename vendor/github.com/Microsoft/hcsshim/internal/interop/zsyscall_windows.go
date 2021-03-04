// Code generated mksyscall_windows.exe DO NOT EDIT

package interop

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
	modapi_ms_win_core_com_l1_1_0 = windows.NewLazySystemDLL("api-ms-win-core-com-l1-1-0.dll")

	procCoTaskMemFree = modapi_ms_win_core_com_l1_1_0.NewProc("CoTaskMemFree")
)

func coTaskMemFree(buffer unsafe.Pointer) {
	syscall.Syscall(procCoTaskMemFree.Addr(), 1, uintptr(buffer), 0, 0)
	return
}
