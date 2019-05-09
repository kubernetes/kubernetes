// +build darwin freebsd openbsd netbsd

package term // import "github.com/docker/docker/pkg/term"

import (
	"unsafe"

	"golang.org/x/sys/unix"
)

const (
	getTermios = unix.TIOCGETA
	setTermios = unix.TIOCSETA
)

// Termios is the Unix API for terminal I/O.
type Termios unix.Termios

// MakeRaw put the terminal connected to the given file descriptor into raw
// mode and returns the previous state of the terminal so that it can be
// restored.
func MakeRaw(fd uintptr) (*State, error) {
	var oldState State
	if _, _, err := unix.Syscall(unix.SYS_IOCTL, fd, getTermios, uintptr(unsafe.Pointer(&oldState.termios))); err != 0 {
		return nil, err
	}

	newState := oldState.termios
	newState.Iflag &^= (unix.IGNBRK | unix.BRKINT | unix.PARMRK | unix.ISTRIP | unix.INLCR | unix.IGNCR | unix.ICRNL | unix.IXON)
	newState.Oflag &^= unix.OPOST
	newState.Lflag &^= (unix.ECHO | unix.ECHONL | unix.ICANON | unix.ISIG | unix.IEXTEN)
	newState.Cflag &^= (unix.CSIZE | unix.PARENB)
	newState.Cflag |= unix.CS8
	newState.Cc[unix.VMIN] = 1
	newState.Cc[unix.VTIME] = 0

	if _, _, err := unix.Syscall(unix.SYS_IOCTL, fd, setTermios, uintptr(unsafe.Pointer(&newState))); err != 0 {
		return nil, err
	}

	return &oldState, nil
}
