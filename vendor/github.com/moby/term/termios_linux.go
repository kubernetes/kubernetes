package term // import "github.com/moby/term"

import (
	"golang.org/x/sys/unix"
)

const (
	getTermios = unix.TCGETS
	setTermios = unix.TCSETS
)

// Termios is the Unix API for terminal I/O.
type Termios unix.Termios

// MakeRaw put the terminal connected to the given file descriptor into raw
// mode and returns the previous state of the terminal so that it can be
// restored.
func MakeRaw(fd uintptr) (*State, error) {
	termios, err := unix.IoctlGetTermios(int(fd), getTermios)
	if err != nil {
		return nil, err
	}

	var oldState State
	oldState.termios = Termios(*termios)

	termios.Iflag &^= (unix.IGNBRK | unix.BRKINT | unix.PARMRK | unix.ISTRIP | unix.INLCR | unix.IGNCR | unix.ICRNL | unix.IXON)
	termios.Oflag &^= unix.OPOST
	termios.Lflag &^= (unix.ECHO | unix.ECHONL | unix.ICANON | unix.ISIG | unix.IEXTEN)
	termios.Cflag &^= (unix.CSIZE | unix.PARENB)
	termios.Cflag |= unix.CS8
	termios.Cc[unix.VMIN] = 1
	termios.Cc[unix.VTIME] = 0

	if err := unix.IoctlSetTermios(int(fd), setTermios, termios); err != nil {
		return nil, err
	}
	return &oldState, nil
}
