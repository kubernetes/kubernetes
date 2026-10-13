//go:build !windows
// +build !windows

package term

import (
	"golang.org/x/sys/unix"
)

// Termios is the Unix API for terminal I/O.
//
// Deprecated: use [unix.Termios].
type Termios = unix.Termios

func makeRaw(fd uintptr) (*State, error) {
	termios, err := tcget(fd)
	if err != nil {
		return nil, err
	}

	oldState := State{termios: *termios}

	termios.Iflag &^= unix.IGNBRK | unix.BRKINT | unix.PARMRK | unix.ISTRIP | unix.INLCR | unix.IGNCR | unix.ICRNL | unix.IXON
	termios.Oflag &^= unix.OPOST
	termios.Lflag &^= unix.ECHO | unix.ECHONL | unix.ICANON | unix.ISIG | unix.IEXTEN
	termios.Cflag &^= unix.CSIZE | unix.PARENB
	termios.Cflag |= unix.CS8
	termios.Cc[unix.VMIN] = 1
	termios.Cc[unix.VTIME] = 0

	if err := tcset(fd, termios); err != nil {
		return nil, err
	}
	return &oldState, nil
}
