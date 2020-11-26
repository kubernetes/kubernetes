// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package socket

import "golang.org/x/sys/unix"

func controlHeaderLen() int {
	return unix.CmsgLen(0)
}

func controlMessageLen(dataLen int) int {
	return unix.CmsgLen(dataLen)
}

func controlMessageSpace(dataLen int) int {
	return unix.CmsgSpace(dataLen)
}
