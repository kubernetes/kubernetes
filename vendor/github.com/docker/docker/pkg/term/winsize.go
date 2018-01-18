// +build !solaris,!windows

package term

import (
	"unsafe"

	"golang.org/x/sys/unix"
)

// GetWinsize returns the window size based on the specified file descriptor.
func GetWinsize(fd uintptr) (*Winsize, error) {
	ws := &Winsize{}
	_, _, err := unix.Syscall(unix.SYS_IOCTL, fd, uintptr(unix.TIOCGWINSZ), uintptr(unsafe.Pointer(ws)))
	// Skipp errno = 0
	if err == 0 {
		return ws, nil
	}
	return ws, err
}

// SetWinsize tries to set the specified window size for the specified file descriptor.
func SetWinsize(fd uintptr, ws *Winsize) error {
	_, _, err := unix.Syscall(unix.SYS_IOCTL, fd, uintptr(unix.TIOCSWINSZ), uintptr(unsafe.Pointer(ws)))
	// Skipp errno = 0
	if err == 0 {
		return nil
	}
	return err
}
