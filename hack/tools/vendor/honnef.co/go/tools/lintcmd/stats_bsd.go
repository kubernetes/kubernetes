// +build darwin dragonfly freebsd netbsd openbsd

package lintcmd

import (
	"os"
	"syscall"
)

var infoSignals = []os.Signal{syscall.SIGINFO}
