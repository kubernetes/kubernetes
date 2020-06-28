package interop

import (
	"syscall"
	"unsafe"
)

//go:generate go run ../../mksyscall_windows.go -output zsyscall_windows.go interop.go

//sys coTaskMemFree(buffer unsafe.Pointer) = api_ms_win_core_com_l1_1_0.CoTaskMemFree

func ConvertAndFreeCoTaskMemString(buffer *uint16) string {
	str := syscall.UTF16ToString((*[1 << 29]uint16)(unsafe.Pointer(buffer))[:])
	coTaskMemFree(unsafe.Pointer(buffer))
	return str
}

func Win32FromHresult(hr uintptr) syscall.Errno {
	if hr&0x1fff0000 == 0x00070000 {
		return syscall.Errno(hr & 0xffff)
	}
	return syscall.Errno(hr)
}
