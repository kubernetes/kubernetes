package testscript

import "golang.org/x/sys/unix"

// cloneFile makes a clone of a file via MacOS's `clonefile` syscall.
func cloneFile(from, to string) error {
	return unix.Clonefile(from, to, 0)
}
