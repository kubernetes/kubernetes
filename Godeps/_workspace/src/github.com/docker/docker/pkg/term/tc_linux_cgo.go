// +build linux,cgo

package term

import (
	"syscall"
	"unsafe"
)

// #include <termios.h>
import "C"

// Termios is the Unix API for terminal I/O.
// It is passthgrouh for syscall.Termios in order to make it portable with
// other platforms where it is not available or handled differently.
type Termios syscall.Termios

// MakeRaw put the terminal connected to the given file descriptor into raw
// mode and returns the previous state of the terminal so that it can be
// restored.
func MakeRaw(fd uintptr) (*State, error) {
	var oldState State
	if err := tcget(fd, &oldState.termios); err != 0 {
		return nil, err
	}

	newState := oldState.termios

	C.cfmakeraw((*C.struct_termios)(unsafe.Pointer(&newState)))
	newState.Oflag = newState.Oflag | C.OPOST
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
