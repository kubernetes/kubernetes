package term

import (
	"syscall"
	"unsafe"
)

const (
	getTermios = syscall.TIOCGETA
	setTermios = syscall.TIOCSETA
)

// Termios magic numbers, passthrough to the ones defined in syscall.
const (
	IGNBRK = syscall.IGNBRK
	PARMRK = syscall.PARMRK
	INLCR  = syscall.INLCR
	IGNCR  = syscall.IGNCR
	ECHONL = syscall.ECHONL
	CSIZE  = syscall.CSIZE
	ICRNL  = syscall.ICRNL
	ISTRIP = syscall.ISTRIP
	PARENB = syscall.PARENB
	ECHO   = syscall.ECHO
	ICANON = syscall.ICANON
	ISIG   = syscall.ISIG
	IXON   = syscall.IXON
	BRKINT = syscall.BRKINT
	INPCK  = syscall.INPCK
	OPOST  = syscall.OPOST
	CS8    = syscall.CS8
	IEXTEN = syscall.IEXTEN
)

// Termios is the Unix API for terminal I/O.
type Termios struct {
	Iflag  uint32
	Oflag  uint32
	Cflag  uint32
	Lflag  uint32
	Cc     [20]byte
	Ispeed uint32
	Ospeed uint32
}

// MakeRaw put the terminal connected to the given file descriptor into raw
// mode and returns the previous state of the terminal so that it can be
// restored.
func MakeRaw(fd uintptr) (*State, error) {
	var oldState State
	if _, _, err := syscall.Syscall(syscall.SYS_IOCTL, fd, uintptr(getTermios), uintptr(unsafe.Pointer(&oldState.termios))); err != 0 {
		return nil, err
	}

	newState := oldState.termios
	newState.Iflag &^= (IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON)
	newState.Oflag &^= OPOST
	newState.Lflag &^= (ECHO | ECHONL | ICANON | ISIG | IEXTEN)
	newState.Cflag &^= (CSIZE | PARENB)
	newState.Cflag |= CS8
	newState.Cc[syscall.VMIN] = 1
	newState.Cc[syscall.VTIME] = 0

	if _, _, err := syscall.Syscall(syscall.SYS_IOCTL, fd, uintptr(setTermios), uintptr(unsafe.Pointer(&newState))); err != 0 {
		return nil, err
	}

	return &oldState, nil
}
