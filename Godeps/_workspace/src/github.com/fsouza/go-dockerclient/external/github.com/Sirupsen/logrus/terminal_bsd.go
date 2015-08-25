// +build darwin freebsd openbsd netbsd dragonfly

package logrus

import "syscall"

const ioctlReadTermios = syscall.TIOCGETA

type Termios syscall.Termios
