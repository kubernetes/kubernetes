// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || linux || netbsd || openbsd || solaris
// +build dragonfly freebsd linux netbsd openbsd solaris

package unix_test

import (
	"testing"

	"golang.org/x/sys/unix"
)

func TestPipe2(t *testing.T) {
	const s = "hello"
	var pipes [2]int
	err := unix.Pipe2(pipes[:], 0)
	if err != nil {
		t.Fatalf("pipe2: %v", err)
	}
	r := pipes[0]
	w := pipes[1]
	go func() {
		n, err := unix.Write(w, []byte(s))
		if err != nil {
			t.Errorf("bad write: %v", err)
			return
		}
		if n != len(s) {
			t.Errorf("bad write count: %d", n)
			return
		}
		err = unix.Close(w)
		if err != nil {
			t.Errorf("bad close: %v", err)
			return
		}
	}()
	var buf [10 + len(s)]byte
	n, err := unix.Read(r, buf[:])
	if err != nil {
		t.Fatalf("bad read: %v", err)
	}
	if n != len(s) {
		t.Fatalf("bad read count: %d", n)
	}
	if string(buf[:n]) != s {
		t.Fatalf("bad contents: %s", string(buf[:n]))
	}
	err = unix.Close(r)
	if err != nil {
		t.Fatalf("bad close: %v", err)
	}
}

func checkNonblocking(t *testing.T, fd int, name string) {
	t.Helper()
	flags, err := unix.FcntlInt(uintptr(fd), unix.F_GETFL, 0)
	if err != nil {
		t.Errorf("fcntl(%s, F_GETFL) failed: %v", name, err)
	} else if flags&unix.O_NONBLOCK == 0 {
		t.Errorf("O_NONBLOCK not set in %s flags %#x", name, flags)
	}
}

func checkCloseonexec(t *testing.T, fd int, name string) {
	t.Helper()
	flags, err := unix.FcntlInt(uintptr(fd), unix.F_GETFD, 0)
	if err != nil {
		t.Errorf("fcntl(%s, F_GETFD) failed: %v", name, err)
	} else if flags&unix.FD_CLOEXEC == 0 {
		t.Errorf("FD_CLOEXEC not set in %s flags %#x", name, flags)
	}
}

func TestNonblockingPipe2(t *testing.T) {
	var pipes [2]int
	err := unix.Pipe2(pipes[:], unix.O_NONBLOCK|unix.O_CLOEXEC)
	if err != nil {
		t.Fatalf("pipe2: %v", err)
	}
	r := pipes[0]
	w := pipes[1]
	defer func() {
		unix.Close(r)
		unix.Close(w)
	}()

	checkNonblocking(t, r, "reader")
	checkCloseonexec(t, r, "reader")
	checkNonblocking(t, w, "writer")
	checkCloseonexec(t, w, "writer")
}
