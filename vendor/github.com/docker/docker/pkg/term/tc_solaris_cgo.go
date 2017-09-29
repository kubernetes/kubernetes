// +build solaris,cgo

package term

import (
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

// #include <termios.h>
import "C"

// Termios is the Unix API for terminal I/O.
// It is passthrough for unix.Termios in order to make it portable with
// other platforms where it is not available or handled differently.
type Termios unix.Termios

// MakeRaw put the terminal connected to the given file descriptor into raw
// mode and returns the previous state of the terminal so that it can be
// restored.
func MakeRaw(fd uintptr) (*State, error) {
	var oldState State
	if err := tcget(fd, &oldState.termios); err != 0 {
		return nil, err
	}

	newState := oldState.termios

	newState.Iflag &^= (unix.IGNBRK | unix.BRKINT | unix.PARMRK | unix.ISTRIP | unix.INLCR | unix.IGNCR | unix.ICRNL | unix.IXON | unix.IXANY)
	newState.Oflag &^= unix.OPOST
	newState.Lflag &^= (unix.ECHO | unix.ECHONL | unix.ICANON | unix.ISIG | unix.IEXTEN)
	newState.Cflag &^= (unix.CSIZE | unix.PARENB)
	newState.Cflag |= unix.CS8

	/*
		VMIN is the minimum number of characters that needs to be read in non-canonical mode for it to be returned
		Since VMIN is overloaded with another element in canonical mode when we switch modes it defaults to 4. It
		needs to be explicitly set to 1.
	*/
	newState.Cc[C.VMIN] = 1
	newState.Cc[C.VTIME] = 0

	if err := tcset(fd, &newState); err != 0 {
		return nil, err
	}
	return &oldState, nil
}

func tcget(fd uintptr, p *Termios) syscall.Errno {
	ret, err := C.tcgetattr(C.int(fd), (*C.struct_termios)(unsafe.Pointer(p)))
	if ret != 0 {
		return err.(syscall.Errno)
	}
	return 0
}

func tcset(fd uintptr, p *Termios) syscall.Errno {
	ret, err := C.tcsetattr(C.int(fd), C.TCSANOW, (*C.struct_termios)(unsafe.Pointer(p)))
	if ret != 0 {
		return err.(syscall.Errno)
	}
	return 0
}
