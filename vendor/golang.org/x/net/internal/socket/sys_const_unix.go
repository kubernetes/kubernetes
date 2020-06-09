// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package socket

import "golang.org/x/sys/unix"

const (
	sysAF_UNSPEC = unix.AF_UNSPEC
	sysAF_INET   = unix.AF_INET
	sysAF_INET6  = unix.AF_INET6

	sysSOCK_RAW = unix.SOCK_RAW
)
