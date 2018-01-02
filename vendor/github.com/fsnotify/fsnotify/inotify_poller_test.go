// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package fsnotify

import (
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

type testFd [2]int

func makeTestFd(t *testing.T) testFd {
	var tfd testFd
	errno := unix.Pipe(tfd[:])
	if errno != nil {
		t.Fatalf("Failed to create pipe: %v", errno)
	}
	return tfd
}

func (tfd testFd) fd() int {
	return tfd[0]
}

func (tfd testFd) closeWrite(t *testing.T) {
	errno := unix.Close(tfd[1])
	if errno != nil {
		t.Fatalf("Failed to close write end of pipe: %v", errno)
	}
}

func (tfd testFd) put(t *testing.T) {
	buf := make([]byte, 10)
	_, errno := unix.Write(tfd[1], buf)
	if errno != nil {
		t.Fatalf("Failed to write to pipe: %v", errno)
	}
}

func (tfd testFd) get(t *testing.T) {
	buf := make([]byte, 10)
	_, errno := unix.Read(tfd[0], buf)
	if errno != nil {
		t.Fatalf("Failed to read from pipe: %v", errno)
	}
}

func (tfd testFd) close() {
	unix.Close(tfd[1])
	unix.Close(tfd[0])
}

func makePoller(t *testing.T) (testFd, *fdPoller) {
	tfd := makeTestFd(t)
	poller, err := newFdPoller(tfd.fd())
	if err != nil {
		t.Fatalf("Failed to create poller: %v", err)
	}
	return tfd, poller
}

func TestPollerWithBadFd(t *testing.T) {
	_, err := newFdPoller(-1)
	if err != unix.EBADF {
		t.Fatalf("Expected EBADF, got: %v", err)
	}
}

func TestPollerWithData(t *testing.T) {
	tfd, poller := makePoller(t)
	defer tfd.close()
	defer poller.close()

	tfd.put(t)
	ok, err := poller.wait()
	if err != nil {
		t.Fatalf("poller failed: %v", err)
	}
	if !ok {
		t.Fatalf("expected poller to return true")
	}
	tfd.get(t)
}

func TestPollerWithWakeup(t *testing.T) {
	tfd, poller := makePoller(t)
	defer tfd.close()
	defer poller.close()

	err := poller.wake()
	if err != nil {
		t.Fatalf("wake failed: %v", err)
	}
	ok, err := poller.wait()
	if err != nil {
		t.Fatalf("poller failed: %v", err)
	}
	if ok {
		t.Fatalf("expected poller to return false")
	}
}

func TestPollerWithClose(t *testing.T) {
	tfd, poller := makePoller(t)
	defer tfd.close()
	defer poller.close()

	tfd.closeWrite(t)
	ok, err := poller.wait()
	if err != nil {
		t.Fatalf("poller failed: %v", err)
	}
	if !ok {
		t.Fatalf("expected poller to return true")
	}
}

func TestPollerWithWakeupAndData(t *testing.T) {
	tfd, poller := makePoller(t)
	defer tfd.close()
	defer poller.close()

	tfd.put(t)
	err := poller.wake()
	if err != nil {
		t.Fatalf("wake failed: %v", err)
	}

	// both data and wakeup
	ok, err := poller.wait()
	if err != nil {
		t.Fatalf("poller failed: %v", err)
	}
	if !ok {
		t.Fatalf("expected poller to return true")
	}

	// data is still in the buffer, wakeup is cleared
	ok, err = poller.wait()
	if err != nil {
		t.Fatalf("poller failed: %v", err)
	}
	if !ok {
		t.Fatalf("expected poller to return true")
	}

	tfd.get(t)
	// data is gone, only wakeup now
	err = poller.wake()
	if err != nil {
		t.Fatalf("wake failed: %v", err)
	}
	ok, err = poller.wait()
	if err != nil {
		t.Fatalf("poller failed: %v", err)
	}
	if ok {
		t.Fatalf("expected poller to return false")
	}
}

func TestPollerConcurrent(t *testing.T) {
	tfd, poller := makePoller(t)
	defer tfd.close()
	defer poller.close()

	oks := make(chan bool)
	live := make(chan bool)
	defer close(live)
	go func() {
		defer close(oks)
		for {
			ok, err := poller.wait()
			if err != nil {
				t.Fatalf("poller failed: %v", err)
			}
			oks <- ok
			if !<-live {
				return
			}
		}
	}()

	// Try a write
	select {
	case <-time.After(50 * time.Millisecond):
	case <-oks:
		t.Fatalf("poller did not wait")
	}
	tfd.put(t)
	if !<-oks {
		t.Fatalf("expected true")
	}
	tfd.get(t)
	live <- true

	// Try a wakeup
	select {
	case <-time.After(50 * time.Millisecond):
	case <-oks:
		t.Fatalf("poller did not wait")
	}
	err := poller.wake()
	if err != nil {
		t.Fatalf("wake failed: %v", err)
	}
	if <-oks {
		t.Fatalf("expected false")
	}
	live <- true

	// Try a close
	select {
	case <-time.After(50 * time.Millisecond):
	case <-oks:
		t.Fatalf("poller did not wait")
	}
	tfd.closeWrite(t)
	if !<-oks {
		t.Fatalf("expected true")
	}
	tfd.get(t)
}
