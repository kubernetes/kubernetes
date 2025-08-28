//go:build !darwin && !freebsd && !netbsd && !openbsd && !js && !wasip1 && !windows
// +build !darwin,!freebsd,!netbsd,!openbsd,!js,!wasip1,!windows

package term

import (
	"golang.org/x/sys/unix"
)

const (
	getTermios = unix.TCGETS
	setTermios = unix.TCSETS
)
