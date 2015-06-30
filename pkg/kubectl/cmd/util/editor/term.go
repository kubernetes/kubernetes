// +build !windows

package editor

import (
	"os"
	"syscall"
)

// childSignals are the allowed signals that can be sent to children in Unix variant OS's
var childSignals = []os.Signal{syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT}
