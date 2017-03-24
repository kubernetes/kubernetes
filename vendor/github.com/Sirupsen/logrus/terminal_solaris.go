// +build solaris,!appengine

package logrus

import (
	"os"

	"golang.org/x/sys/unix"
)

// IsTerminal returns true if the given file descriptor is a terminal.
func IsTerminal() bool {
	_, err := unix.IoctlGetTermios(int(os.Stdout.Fd()), unix.TCGETA)
	return err == nil
}
