// +build solaris,cgo

package term

import (
	"unsafe"

	"golang.org/x/sys/unix"
)

/*
#include <unistd.h>
#include <stropts.h>
#include <termios.h>

// Small wrapper to get rid of variadic args of ioctl()
int my_ioctl(int fd, int cmd, struct winsize *ws) {
	return ioctl(fd, cmd, ws);
}
*/
import "C"

// GetWinsize returns the window size based on the specified file descriptor.
func GetWinsize(fd uintptr) (*Winsize, error) {
	ws := &Winsize{}
	ret, err := C.my_ioctl(C.int(fd), C.int(unix.TIOCGWINSZ), (*C.struct_winsize)(unsafe.Pointer(ws)))
	// Skip retval = 0
	if ret == 0 {
		return ws, nil
	}
	return ws, err
}

// SetWinsize tries to set the specified window size for the specified file descriptor.
func SetWinsize(fd uintptr, ws *Winsize) error {
	ret, err := C.my_ioctl(C.int(fd), C.int(unix.TIOCSWINSZ), (*C.struct_winsize)(unsafe.Pointer(ws)))
	// Skip retval = 0
	if ret == 0 {
		return nil
	}
	return err
}
