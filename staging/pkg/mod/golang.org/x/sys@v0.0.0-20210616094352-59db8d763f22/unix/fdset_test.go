// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris
// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package unix_test

import (
	"testing"

	"golang.org/x/sys/unix"
)

func TestFdSet(t *testing.T) {
	var fdSet unix.FdSet
	fdSet.Zero()
	for fd := 0; fd < unix.FD_SETSIZE; fd++ {
		if fdSet.IsSet(fd) {
			t.Fatalf("Zero did not clear fd %d", fd)
		}
		fdSet.Set(fd)
	}

	for fd := 0; fd < unix.FD_SETSIZE; fd++ {
		if !fdSet.IsSet(fd) {
			t.Fatalf("IsSet(%d): expected true, got false", fd)
		}
	}

	fdSet.Zero()
	for fd := 0; fd < unix.FD_SETSIZE; fd++ {
		if fdSet.IsSet(fd) {
			t.Fatalf("Zero did not clear fd %d", fd)
		}
	}

	for fd := 1; fd < unix.FD_SETSIZE; fd += 2 {
		fdSet.Set(fd)
	}

	for fd := 0; fd < unix.FD_SETSIZE; fd++ {
		if fd&0x1 == 0x1 {
			if !fdSet.IsSet(fd) {
				t.Fatalf("IsSet(%d): expected true, got false", fd)
			}
		} else {
			if fdSet.IsSet(fd) {
				t.Fatalf("IsSet(%d): expected false, got true", fd)
			}
		}
	}

	for fd := 1; fd < unix.FD_SETSIZE; fd += 2 {
		fdSet.Clear(fd)
	}

	for fd := 0; fd < unix.FD_SETSIZE; fd++ {
		if fdSet.IsSet(fd) {
			t.Fatalf("Clear(%d) did not clear fd", fd)
		}
	}
}
