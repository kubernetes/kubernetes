//go:build !go1.20
// +build !go1.20

package libcontainer

import "golang.org/x/sys/unix"

func eaccess(path string) error {
	// This check is similar to access(2) with X_OK except for
	// setuid/setgid binaries where it checks against the effective
	// (rather than real) uid and gid. It is not needed in go 1.20
	// and beyond and will be removed later.

	// Relies on code added in https://go-review.googlesource.com/c/sys/+/468877
	// and older CLs linked from there.
	return unix.Faccessat(unix.AT_FDCWD, path, unix.X_OK, unix.AT_EACCESS)
}
