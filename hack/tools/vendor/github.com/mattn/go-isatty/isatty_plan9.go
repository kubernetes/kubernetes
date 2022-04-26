//go:build plan9
// +build plan9

package isatty

import (
	"syscall"
)

// IsTerminal returns true if the given file descriptor is a terminal.
func IsTerminal(fd uintptr) bool {
	path, err := syscall.Fd2path(int(fd))
	if err != nil {
		return false
	}
	return path == "/dev/cons" || path == "/mnt/term/dev/cons"
}

// IsCygwinTerminal return true if the file descriptor is a cygwin or msys2
// terminal. This is also always false on this environment.
func IsCygwinTerminal(fd uintptr) bool {
	return false
}
