// +build aix android linux solaris

package lintcmd

import (
	"os"
	"syscall"
)

var infoSignals = []os.Signal{syscall.SIGUSR1}
