//go:build solaris && !appengine
// +build solaris,!appengine

package isatty

import (
	"golang.org/x/sys/unix"
)

// IsTerminal returns true if the given file descriptor is a terminal.
// see: https://src.illumos.org/source/xref/illumos-gate/usr/src/lib/libc/port/gen/isatty.c
func IsTerminal(fd uintptr) bool {
	_, err := unix.IoctlGetTermio(int(fd), unix.TCGETA)
	return err == nil
}

// IsCygwinTerminal return true if the file descriptor is a cygwin or msys2
// terminal. This is also always false on this environment.
func IsCygwinTerminal(fd uintptr) bool {
	return false
}
