// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package fsnotify

import (
	"errors"
	"syscall"
)

type fdPoller struct {
	fd   int    // File descriptor (as returned by the inotify_init() syscall)
	epfd int    // Epoll file descriptor
	pipe [2]int // Pipe for waking up
}

func emptyPoller(fd int) *fdPoller {
	poller := new(fdPoller)
	poller.fd = fd
	poller.epfd = -1
	poller.pipe[0] = -1
	poller.pipe[1] = -1
	return poller
}

// Create a new inotify poller.
// This creates an inotify handler, and an epoll handler.
func newFdPoller(fd int) (*fdPoller, error) {
	var errno error
	poller := emptyPoller(fd)
	defer func() {
		if errno != nil {
			poller.close()
		}
	}()
	poller.fd = fd

	// Create epoll fd
	poller.epfd, errno = syscall.EpollCreate(1)
	if poller.epfd == -1 {
		return nil, errno
	}
	// Create pipe; pipe[0] is the read end, pipe[1] the write end.
	errno = syscall.Pipe2(poller.pipe[:], syscall.O_NONBLOCK)
	if errno != nil {
		return nil, errno
	}

	// Register inotify fd with epoll
	event := syscall.EpollEvent{
		Fd:     int32(poller.fd),
		Events: syscall.EPOLLIN,
	}
	errno = syscall.EpollCtl(poller.epfd, syscall.EPOLL_CTL_ADD, poller.fd, &event)
	if errno != nil {
		return nil, errno
	}

	// Register pipe fd with epoll
	event = syscall.EpollEvent{
		Fd:     int32(poller.pipe[0]),
		Events: syscall.EPOLLIN,
	}
	errno = syscall.EpollCtl(poller.epfd, syscall.EPOLL_CTL_ADD, poller.pipe[0], &event)
	if errno != nil {
		return nil, errno
	}

	return poller, nil
}

// Wait using epoll.
// Returns true if something is ready to be read,
// false if there is not.
func (poller *fdPoller) wait() (bool, error) {
	// 3 possible events per fd, and 2 fds, makes a maximum of 6 events.
	// I don't know whether epoll_wait returns the number of events returned,
	// or the total number of events ready.
	// I decided to catch both by making the buffer one larger than the maximum.
	events := make([]syscall.EpollEvent, 7)
	for {
		n, errno := syscall.EpollWait(poller.epfd, events, -1)
		if n == -1 {
			if errno == syscall.EINTR {
				continue
			}
			return false, errno
		}
		if n == 0 {
			// If there are no events, try again.
			continue
		}
		if n > 6 {
			// This should never happen. More events were returned than should be possible.
			return false, errors.New("epoll_wait returned more events than I know what to do with")
		}
		ready := events[:n]
		epollhup := false
		epollerr := false
		epollin := false
		for _, event := range ready {
			if event.Fd == int32(poller.fd) {
				if event.Events&syscall.EPOLLHUP != 0 {
					// This should not happen, but if it does, treat it as a wakeup.
					epollhup = true
				}
				if event.Events&syscall.EPOLLERR != 0 {
					// If an error is waiting on the file descriptor, we should pretend
					// something is ready to read, and let syscall.Read pick up the error.
					epollerr = true
				}
				if event.Events&syscall.EPOLLIN != 0 {
					// There is data to read.
					epollin = true
				}
			}
			if event.Fd == int32(poller.pipe[0]) {
				if event.Events&syscall.EPOLLHUP != 0 {
					// Write pipe descriptor was closed, by us. This means we're closing down the
					// watcher, and we should wake up.
				}
				if event.Events&syscall.EPOLLERR != 0 {
					// If an error is waiting on the pipe file descriptor.
					// This is an absolute mystery, and should never ever happen.
					return false, errors.New("Error on the pipe descriptor.")
				}
				if event.Events&syscall.EPOLLIN != 0 {
					// This is a regular wakeup, so we have to clear the buffer.
					err := poller.clearWake()
					if err != nil {
						return false, err
					}
				}
			}
		}

		if epollhup || epollerr || epollin {
			return true, nil
		}
		return false, nil
	}
}

// Close the write end of the poller.
func (poller *fdPoller) wake() error {
	buf := make([]byte, 1)
	n, errno := syscall.Write(poller.pipe[1], buf)
	if n == -1 {
		if errno == syscall.EAGAIN {
			// Buffer is full, poller will wake.
			return nil
		}
		return errno
	}
	return nil
}

func (poller *fdPoller) clearWake() error {
	// You have to be woken up a LOT in order to get to 100!
	buf := make([]byte, 100)
	n, errno := syscall.Read(poller.pipe[0], buf)
	if n == -1 {
		if errno == syscall.EAGAIN {
			// Buffer is empty, someone else cleared our wake.
			return nil
		}
		return errno
	}
	return nil
}

// Close all poller file descriptors, but not the one passed to it.
func (poller *fdPoller) close() {
	if poller.pipe[1] != -1 {
		syscall.Close(poller.pipe[1])
	}
	if poller.pipe[0] != -1 {
		syscall.Close(poller.pipe[0])
	}
	if poller.epfd != -1 {
		syscall.Close(poller.epfd)
	}
}
