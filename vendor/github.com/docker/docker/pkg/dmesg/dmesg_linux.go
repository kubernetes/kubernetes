package dmesg

import (
	"unsafe"

	"golang.org/x/sys/unix"
)

// Dmesg returns last messages from the kernel log, up to size bytes
func Dmesg(size int) []byte {
	t := uintptr(3) // SYSLOG_ACTION_READ_ALL
	b := make([]byte, size)
	amt, _, err := unix.Syscall(unix.SYS_SYSLOG, t, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)))
	if err != 0 {
		return []byte{}
	}
	return b[:amt]
}
