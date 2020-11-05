// +build !windows

package term

import (
	"golang.org/x/sys/unix"
)

func tcget(fd uintptr) (*Termios, error) {
	p, err := unix.IoctlGetTermios(int(fd), getTermios)
	if err != nil {
		return nil, err
	}
	return (*Termios)(p), nil
}

func tcset(fd uintptr, p *Termios) error {
	return unix.IoctlSetTermios(int(fd), setTermios, (*unix.Termios)(p))
}
