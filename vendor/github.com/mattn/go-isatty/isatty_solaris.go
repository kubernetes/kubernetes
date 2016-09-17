// +build solaris
// +build !appengine

package isatty

import (
	"golang.org/x/sys/unix"
)

// IsTerminal returns true if the given file descriptor is a terminal.
// see: http://src.illumos.org/source/xref/illumos-gate/usr/src/lib/libbc/libc/gen/common/isatty.c
func IsTerminal(fd uintptr) bool {
	var termio unix.Termio
	err := unix.IoctlSetTermio(int(fd), unix.TCGETA, &termio)
	return err == nil
}
