// +build solaris,!appengine

package logrus

import (
	"io"
	"os"

	"golang.org/x/sys/unix"
)

// IsTerminal returns true if the given file descriptor is a terminal.
func IsTerminal(f io.Writer) bool {
	switch v := f.(type) {
	case *os.File:
		_, err := unix.IoctlGetTermios(int(v.Fd()), unix.TCGETA)
		return err == nil
	default:
		return false
	}
}
