// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package socket

import "syscall"

var (
	errERROR_IO_PENDING error = syscall.ERROR_IO_PENDING
	errEINVAL           error = syscall.EINVAL
)

// errnoErr returns common boxed Errno values, to prevent allocations
// at runtime.
func errnoErr(errno syscall.Errno) error {
	switch errno {
	case 0:
		return nil
	case syscall.ERROR_IO_PENDING:
		return errERROR_IO_PENDING
	case syscall.EINVAL:
		return errEINVAL
	}
	return errno
}
