//go:build darwin || freebsd || openbsd || netbsd
// +build darwin freebsd openbsd netbsd

package term

import (
	"golang.org/x/sys/unix"
)

const (
	getTermios = unix.TIOCGETA
	setTermios = unix.TIOCSETA
)
