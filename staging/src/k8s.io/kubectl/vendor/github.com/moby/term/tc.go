//go:build !windows
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
	return p, nil
}

func tcset(fd uintptr, p *Termios) error {
	return unix.IoctlSetTermios(int(fd), setTermios, p)
}
