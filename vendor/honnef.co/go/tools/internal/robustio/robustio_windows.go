// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package robustio

import (
	"os"
	"syscall"
)

const errFileNotFound = syscall.ERROR_FILE_NOT_FOUND

// isEphemeralError returns true if err may be resolved by waiting.
func isEphemeralError(err error) bool {
	switch werr := err.(type) {
	case *os.PathError:
		err = werr.Err
	case *os.LinkError:
		err = werr.Err
	case *os.SyscallError:
		err = werr.Err
	}
	if errno, ok := err.(syscall.Errno); ok {
		switch errno {
		case syscall.ERROR_ACCESS_DENIED,
			syscall.ERROR_FILE_NOT_FOUND,
			ERROR_SHARING_VIOLATION:
			return true
		}
	}
	return false
}
