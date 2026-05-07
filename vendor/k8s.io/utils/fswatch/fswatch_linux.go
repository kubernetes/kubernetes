/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package fswatch

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"
)

const portableDefaultMask = syscall.IN_CREATE |
	syscall.IN_MOVED_TO |
	syscall.IN_MODIFY |
	syscall.IN_ATTRIB |
	syscall.IN_DELETE |
	syscall.IN_DELETE_SELF |
	syscall.IN_MOVE_SELF |
	syscall.IN_MOVED_FROM

// inotifyImpl is the Linux watcher backed by raw inotify(7), epoll for
// shutdown multiplexing, and a pipe for shutdown wakeup. It deliberately
// keeps no fsnotify dependency so the Linux build closure stays clean.
type inotifyImpl struct {
	fd     int
	epollF int
	wakeRd int
	wakeWr int

	exited       chan struct{}
	done         chan struct{}
	shuttingDown atomic.Bool

	mu      sync.Mutex
	closed  bool
	watches map[uint32]string
	paths   map[string]uint32

	events chan Event
	errors chan error
}

func newWatcherImpl() (watcherImpl, error) {
	fd, err := syscall.InotifyInit1(syscall.IN_CLOEXEC | syscall.IN_NONBLOCK)
	if err != nil {
		return nil, fmt.Errorf("inotify_init1: %w", err)
	}

	var pipeFds [2]int
	if err := syscall.Pipe2(pipeFds[:], syscall.O_CLOEXEC|syscall.O_NONBLOCK); err != nil {
		syscall.Close(fd)
		return nil, fmt.Errorf("pipe2: %w", err)
	}

	epollF, err := syscall.EpollCreate1(syscall.EPOLL_CLOEXEC)
	if err != nil {
		syscall.Close(fd)
		syscall.Close(pipeFds[0])
		syscall.Close(pipeFds[1])
		return nil, fmt.Errorf("epoll_create1: %w", err)
	}

	if err := syscall.EpollCtl(epollF, syscall.EPOLL_CTL_ADD, fd,
		&syscall.EpollEvent{Events: syscall.EPOLLIN, Fd: int32(fd)}); err != nil {
		syscall.Close(fd)
		syscall.Close(pipeFds[0])
		syscall.Close(pipeFds[1])
		syscall.Close(epollF)
		return nil, fmt.Errorf("epoll_ctl(inotify): %w", err)
	}
	if err := syscall.EpollCtl(epollF, syscall.EPOLL_CTL_ADD, pipeFds[0],
		&syscall.EpollEvent{Events: syscall.EPOLLIN, Fd: int32(pipeFds[0])}); err != nil {
		syscall.Close(fd)
		syscall.Close(pipeFds[0])
		syscall.Close(pipeFds[1])
		syscall.Close(epollF)
		return nil, fmt.Errorf("epoll_ctl(wakeRd): %w", err)
	}

	b := &inotifyImpl{
		fd:      fd,
		epollF:  epollF,
		wakeRd:  pipeFds[0],
		wakeWr:  pipeFds[1],
		exited:  make(chan struct{}),
		done:    make(chan struct{}),
		watches: make(map[uint32]string),
		paths:   make(map[string]uint32),
		events:  make(chan Event, 64),
		errors:  make(chan error, 64),
	}
	go b.readEvents()
	return b, nil
}

func (b *inotifyImpl) Add(path string) error {
	// Hold mu across the syscall so a concurrent Close cannot close
	// b.fd while we are issuing inotify_add_watch on it. Close waits
	// for the read goroutine to exit before closing FDs anyway, but
	// it sets b.closed under mu first; observing closed=false under
	// mu and keeping the lock proves the FD is still valid.
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return ErrClosed
	}
	wd, err := syscall.InotifyAddWatch(b.fd, path, portableDefaultMask)
	if err != nil {
		return err
	}
	if existing, ok := b.paths[path]; ok && existing != uint32(wd) {
		delete(b.watches, existing)
	}
	b.watches[uint32(wd)] = path
	b.paths[path] = uint32(wd)
	return nil
}

func (b *inotifyImpl) Remove(path string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return ErrClosed
	}
	wd, ok := b.paths[path]
	if !ok {
		return ErrNonExistentWatch
	}
	delete(b.paths, path)
	delete(b.watches, wd)
	if _, err := syscall.InotifyRmWatch(b.fd, wd); err != nil {
		// EINVAL means the kernel already removed the watch (e.g.
		// the watched path was deleted).
		if errors.Is(err, syscall.EINVAL) {
			return nil
		}
		return err
	}
	return nil
}

func (b *inotifyImpl) Events() <-chan Event { return b.events }
func (b *inotifyImpl) Errors() <-chan error { return b.errors }

func (b *inotifyImpl) Close() error {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil
	}
	b.closed = true
	b.mu.Unlock()

	b.shuttingDown.Store(true)
	close(b.done)
	closeErr := syscall.Close(b.wakeWr)
	<-b.exited
	syscall.Close(b.fd)
	syscall.Close(b.epollF)
	syscall.Close(b.wakeRd)
	close(b.events)
	close(b.errors)
	return closeErr
}

func (b *inotifyImpl) readEvents() {
	defer close(b.exited)

	var epollEvents [2]syscall.EpollEvent
	var buf [syscall.SizeofInotifyEvent * 4096]byte

	for {
		n, err := syscall.EpollWait(b.epollF, epollEvents[:], -1)
		if err != nil {
			if errors.Is(err, syscall.EINTR) {
				continue
			}
			b.deliverError(err)
			return
		}
		for i := 0; i < n; i++ {
			switch epollEvents[i].Fd {
			case int32(b.wakeRd):
				return
			case int32(b.fd):
				if !b.drainInotify(buf[:]) {
					return
				}
			}
		}
	}
}

func (b *inotifyImpl) drainInotify(buf []byte) bool {
	for {
		select {
		case <-b.done:
			return false
		default:
		}
		if b.shuttingDown.Load() {
			return false
		}

		n, err := syscall.Read(b.fd, buf)
		if err != nil {
			if errors.Is(err, syscall.EAGAIN) || errors.Is(err, syscall.EWOULDBLOCK) {
				return true
			}
			if errors.Is(err, syscall.EINTR) {
				continue
			}
			b.deliverError(err)
			return false
		}
		if n < syscall.SizeofInotifyEvent {
			b.deliverError(errors.New("fswatch: short inotify read"))
			continue
		}

		var offset uint32
		for offset <= uint32(n)-syscall.SizeofInotifyEvent {
			select {
			case <-b.done:
				return false
			default:
			}
			raw := (*syscall.InotifyEvent)(unsafe.Pointer(&buf[offset]))
			b.deliverInotifyEvent(raw, buf, offset)
			offset += syscall.SizeofInotifyEvent + raw.Len
		}
	}
}

func (b *inotifyImpl) deliverInotifyEvent(raw *syscall.InotifyEvent, buf []byte, offset uint32) {
	if uint32(raw.Mask)&syscall.IN_Q_OVERFLOW != 0 {
		select {
		case b.errors <- ErrEventOverflow:
		default:
		}
		return
	}

	op := translateMaskToOp(uint32(raw.Mask))
	if op == 0 {
		return
	}

	name := b.resolveName(raw, buf, offset)
	ev := Event{Name: name, Op: op}

	select {
	case <-b.done:
		return
	case b.events <- ev:
	}
}

// resolveName builds the absolute event path. When the kernel does not
// append a filename (events targeting the watched path itself), the
// watch's stored path is used.
func (b *inotifyImpl) resolveName(raw *syscall.InotifyEvent, buf []byte, offset uint32) string {
	b.mu.Lock()
	name := b.watches[uint32(raw.Wd)]
	b.mu.Unlock()
	if raw.Len > 0 {
		nameBytes := buf[offset+syscall.SizeofInotifyEvent : offset+syscall.SizeofInotifyEvent+raw.Len]
		for i, c := range nameBytes {
			if c == 0 {
				nameBytes = nameBytes[:i]
				break
			}
		}
		if len(nameBytes) > 0 {
			name = name + "/" + string(nameBytes)
		}
	}
	return name
}

func (b *inotifyImpl) deliverError(err error) {
	select {
	case b.errors <- err:
	default:
	}
}

// translateMaskToOp mirrors fsnotify v1's newEvent translation. Note
// that IN_MOVED_TO maps to Create (file moved into watched dir), not
// Rename.
func translateMaskToOp(mask uint32) Op {
	var op Op
	if mask&syscall.IN_CREATE != 0 || mask&syscall.IN_MOVED_TO != 0 {
		op |= Create
	}
	if mask&syscall.IN_DELETE != 0 || mask&syscall.IN_DELETE_SELF != 0 {
		op |= Remove
	}
	if mask&syscall.IN_MODIFY != 0 {
		op |= Write
	}
	if mask&syscall.IN_MOVE_SELF != 0 || mask&syscall.IN_MOVED_FROM != 0 {
		op |= Rename
	}
	if mask&syscall.IN_ATTRIB != 0 {
		op |= Chmod
	}
	return op
}
