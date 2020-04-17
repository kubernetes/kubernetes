// +build darwin dragonfly freebsd netbsd openbsd

package lintutil

import (
	"os"
	"syscall"
)

var infoSignals = []os.Signal{syscall.SIGINFO}
