package socket

import (
	"unsafe"
)

// RawSockaddr allows structs to be used with [Bind] and [ConnectEx]. The
// struct must meet the Win32 sockaddr requirements specified here:
// https://docs.microsoft.com/en-us/windows/win32/winsock/sockaddr-2
//
// Specifically, the struct size must be least larger than an int16 (unsigned short)
// for the address family.
type RawSockaddr interface {
	// Sockaddr returns a pointer to the RawSockaddr and its struct size, allowing
	// for the RawSockaddr's data to be overwritten by syscalls (if necessary).
	//
	// It is the callers responsibility to validate that the values are valid; invalid
	// pointers or size can cause a panic.
	Sockaddr() (unsafe.Pointer, int32, error)
}
