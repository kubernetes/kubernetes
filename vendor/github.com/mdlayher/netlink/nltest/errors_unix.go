//go:build !plan9 && !windows

package nltest

import (
	"errors"

	"golang.org/x/sys/unix"
)

func isSyscallError(err error) bool {
	var errno unix.Errno
	ok := errors.As(err, &errno)
	return ok
}
