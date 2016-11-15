// +build freebsd openbsd netbsd dragonfly darwin linux solaris

package interrupthandler

import (
	"os"
	"os/signal"
	"syscall"
)

func SwallowSigQuit() {
	c := make(chan os.Signal, 1024)
	signal.Notify(c, syscall.SIGQUIT)
}
