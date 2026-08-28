//go:build !appengine && !js && !windows && !nacl && !plan9 && !wasi && !wasip1 && !tinygo

package logrus

import (
	"io"
	"os"
)

func checkIfTerminal(w io.Writer) bool {
	switch v := w.(type) {
	case *os.File:
		fd := v.Fd()
		if fd > uintptr(^uint(0)>>1) {
			return false
		}
		return isTerminal(int(fd))
	default:
		return false
	}
}
