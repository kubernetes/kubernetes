// +build !appengine,!js,windows

package logrus

import (
	"io"
	"os"
	"syscall"
)

func checkIfTerminal(w io.Writer) bool {
	switch v := w.(type) {
	case *os.File:
		var mode uint32
		err := syscall.GetConsoleMode(syscall.Handle(v.Fd()), &mode)
		return err == nil
	default:
		return false
	}
}
