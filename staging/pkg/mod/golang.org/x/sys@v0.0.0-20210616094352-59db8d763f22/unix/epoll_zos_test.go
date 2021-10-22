// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos && s390x
// +build zos,s390x

package unix_test

// Modified from Linux tests for epoll.

import (
	"os"
	"testing"

	"golang.org/x/sys/unix"
)

func TestEpollIn(t *testing.T) {
	efd, err := unix.EpollCreate1(0) // no CLOEXEC equivalent on z/OS
	if err != nil {
		t.Fatalf("EpollCreate1: %v", err)
	}
	// no need to defer a close on efd, as it's not a real file descriptor on zos

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	fd := int(r.Fd())
	ev := unix.EpollEvent{Events: unix.EPOLLIN, Fd: int32(fd)}

	err = unix.EpollCtl(efd, unix.EPOLL_CTL_ADD, fd, &ev)
	if err != nil {
		t.Fatalf("EpollCtl: %v", err)
	}

	if _, err := w.Write([]byte("HELLO GOPHER")); err != nil {
		t.Fatal(err)
	}

	events := make([]unix.EpollEvent, 128)
	n, err := unix.EpollWait(efd, events, 1)
	if err != nil {
		t.Fatalf("EpollWait: %v", err)
	}

	if n != 1 {
		t.Errorf("EpollWait: wrong number of events: got %v, expected 1", n)
	}

	got := int(events[0].Fd)
	if got != fd {
		t.Errorf("EpollWait: wrong Fd in event: got %v, expected %v", got, fd)
	}

	if events[0].Events&unix.EPOLLIN == 0 {
		t.Errorf("Expected EPOLLIN flag to be set, got %b", events[0].Events)
	}
}

func TestEpollHup(t *testing.T) {
	efd, err := unix.EpollCreate1(0)
	if err != nil {
		t.Fatalf("EpollCreate1: %v", err)
	}
	// no need to defer a close on efd, as it's not a real file descriptor on zos

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	fd := int(r.Fd())
	// EPOLLHUP should be reported even if not explicitly requested
	ev := unix.EpollEvent{Events: unix.EPOLLIN, Fd: int32(fd)}

	err = unix.EpollCtl(efd, unix.EPOLL_CTL_ADD, fd, &ev)
	if err != nil {
		t.Fatalf("EpollCtl: %v", err)
	}

	events := make([]unix.EpollEvent, 128)
	n, err := unix.EpollWait(efd, events, 1)
	if err != nil {
		t.Fatalf("EpollWait: %v", err)
	}

	if events[0].Events&unix.EPOLLHUP != 0 {
		t.Errorf("EPOLLHUP flag aset without hangup event; got n=%d, flags=%b", n, events[0].Events)
	}

	w.Close()

	events = make([]unix.EpollEvent, 128)
	n, err = unix.EpollWait(efd, events, 1)
	if err != nil {
		t.Fatalf("EpollWait: %v", err)
	}

	if n < 1 || events[0].Events&unix.EPOLLHUP == 0 {
		t.Errorf("Expected EPOLLHUP flag to be set, got n=%d, flags=%b", n, events[0].Events)
	}

}

func TestEpollInManyFds(t *testing.T) {
	efd, err := unix.EpollCreate1(4) // Like on Linux, size arg is ignored.
	if err != nil {
		t.Fatalf("EpollCreate: %v", err)
	}
	// no need to defer a close on efd, as it's not a real file descriptor on zos

	rFds := make([]int, 10)
	wPipes := make([]*os.File, 10)

	for i := 0; i < 10; i++ {
		r, w, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()
		defer w.Close()

		rFds[i] = int(r.Fd())
		wPipes[i] = w
	}

	// Monitor all 10 read pipes
	for _, fd := range rFds {
		ev := unix.EpollEvent{Events: unix.EPOLLIN, Fd: int32(fd)}
		err = unix.EpollCtl(efd, unix.EPOLL_CTL_ADD, fd, &ev)
		if err != nil {
			t.Fatalf("EpollCtl: %v", err)
		}
	}

	// Write to only 5 odd-numbered pipes
	for i, w := range wPipes {
		if i%2 == 0 {
			continue
		}
		if _, err := w.Write([]byte("HELLO")); err != nil {
			t.Fatal(err)
		}
	}

	events := make([]unix.EpollEvent, 128)
	n, err := unix.EpollWait(efd, events, 1)
	if err != nil {
		t.Fatalf("EpollWait: %v", err)
	}

	if n != 5 {
		t.Errorf("EpollWait: wrong number of events: got %v, expected 5", n)
	}

	// Check level triggering here
	if _, err := wPipes[0].Write([]byte("HELLO")); err != nil {
		t.Fatal(err)
	}

	// Now, a total of 6 pipes have been written to - level triggered notifis should number 6
	events = make([]unix.EpollEvent, 128)
	n, err = unix.EpollWait(efd, events, 1)
	if err != nil {
		t.Fatalf("EpollWait: %v", err)
	}

	if n != 6 {
		t.Errorf("EpollWait: wrong number of events: got %v, expected 6", n)
	}

}

func TestMultipleEpolls(t *testing.T) {
	efd1, err := unix.EpollCreate1(4)
	if err != nil {
		t.Fatalf("EpollCreate: %v", err)
	}
	// no need to defer a close on efd1, as it's not a real file descriptor on zos

	efd2, err := unix.EpollCreate1(4)
	if err != nil {
		t.Fatalf("EpollCreate: %v", err)
	}
	// no need to defer a close on efd2, as it's not a real file descriptor on zos

	rFds := make([]int, 10)
	wPipes := make([]*os.File, 10)

	for i := 0; i < 10; i++ {
		r, w, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()
		defer w.Close()

		rFds[i] = int(r.Fd())
		wPipes[i] = w
	}

	// Monitor first 7 read pipes on epoll1, last 3 on epoll2
	for i, fd := range rFds {
		var efd int
		if i < 7 {
			efd = efd1
		} else {
			efd = efd2
		}
		ev := unix.EpollEvent{Events: unix.EPOLLIN, Fd: int32(fd)}
		err = unix.EpollCtl(efd, unix.EPOLL_CTL_ADD, fd, &ev)
		if err != nil {
			t.Fatalf("EpollCtl: %v", err)
		}
	}

	// Write to all 10 pipes
	for _, w := range wPipes {
		if _, err := w.Write([]byte("HELLO")); err != nil {
			t.Fatal(err)
		}
	}

	events := make([]unix.EpollEvent, 128)
	n, err := unix.EpollWait(efd1, events, 1)
	if err != nil {
		t.Fatalf("EpollWait: %v", err)
	}
	if n != 7 {
		t.Errorf("EpollWait: wrong number of events on ep1: got %v, expected 7", n)
	}

	events = make([]unix.EpollEvent, 128)
	n, err = unix.EpollWait(efd2, events, 1)
	if err != nil {
		t.Fatalf("EpollWait: %v", err)
	}
	if n != 3 {
		t.Errorf("EpollWait: wrong number of events on ep2: got %v, expected 3", n)
	}
}

func TestEpollErrors(t *testing.T) {
	efd, err := unix.EpollCreate1(4)
	if err != nil {
		t.Fatalf("EpollCreate: %v", err)
	}
	// no need to defer a close on efd, as it's not a real file descriptor on zos

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()

	fd := int(r.Fd())

	ev := unix.EpollEvent{Events: unix.EPOLLIN, Fd: int32(fd)}
	if err = unix.EpollCtl(efd+1, unix.EPOLL_CTL_ADD, fd, &ev); err != unix.EBADF {
		t.Errorf("EpollCtl: got %v when EpollCtl ADD called with invalid epfd, expected EBADF", err)
	}

	if err = unix.EpollCtl(efd, unix.EPOLL_CTL_ADD, fd, &ev); err != nil {
		t.Fatalf("EpollCtl: %v", err)
	}

	if err = unix.EpollCtl(efd, unix.EPOLL_CTL_MOD, -2, &ev); err != unix.ENOENT {
		t.Errorf("EpollCtl: got %v when EpollCtl MOD called with invalid fd, expected ENOENT", err)
	}
}
