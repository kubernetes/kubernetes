// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fsnotify

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"syscall"
	"unsafe"
)

const (
	sys_AGNOSTIC_EVENTS = syscall.IN_MOVED_TO | syscall.IN_MOVED_FROM |
		syscall.IN_CREATE | syscall.IN_ATTRIB | syscall.IN_MODIFY |
		syscall.IN_MOVE_SELF | syscall.IN_DELETE | syscall.IN_DELETE_SELF
)

func newEvent(name string, mask uint32) Event {
	e := Event{Name: name}
	if mask&syscall.IN_CREATE == syscall.IN_CREATE || mask&syscall.IN_MOVED_TO == syscall.IN_MOVED_TO {
		e.Op |= Create
	}
	if mask&syscall.IN_DELETE_SELF == syscall.IN_DELETE_SELF || mask&syscall.IN_DELETE == syscall.IN_DELETE {
		e.Op |= Remove
	}
	if mask&syscall.IN_MODIFY == syscall.IN_MODIFY {
		e.Op |= Write
	}
	if mask&syscall.IN_MOVE_SELF == syscall.IN_MOVE_SELF || mask&syscall.IN_MOVED_FROM == syscall.IN_MOVED_FROM {
		e.Op |= Rename
	}
	if mask&syscall.IN_ATTRIB == syscall.IN_ATTRIB {
		e.Op |= Chmod
	}
	return e
}

type watch struct {
	wd    uint32 // Watch descriptor (as returned by the inotify_add_watch() syscall)
	flags uint32 // inotify flags of this watch (see inotify(7) for the list of valid flags)
}

type Watcher struct {
	mu       sync.Mutex        // Map access
	fd       int               // File descriptor (as returned by the inotify_init() syscall)
	watches  map[string]*watch // Map of inotify watches (key: path)
	paths    map[int]string    // Map of watched paths (key: watch descriptor)
	Errors   chan error        // Errors are sent on this channel
	Events   chan Event        // Events are returned on this channel
	done     chan bool         // Channel for sending a "quit message" to the reader goroutine
	isClosed bool              // Set to true when Close() is first called
}

// NewWatcher creates and returns a new inotify instance using inotify_init(2)
func NewWatcher() (*Watcher, error) {
	fd, errno := syscall.InotifyInit()
	if fd == -1 {
		return nil, os.NewSyscallError("inotify_init", errno)
	}
	w := &Watcher{
		fd:      fd,
		watches: make(map[string]*watch),
		paths:   make(map[int]string),
		Events:  make(chan Event),
		Errors:  make(chan error),
		done:    make(chan bool, 1),
	}

	go w.readEvents()
	return w, nil
}

// Close closes an inotify watcher instance
// It sends a message to the reader goroutine to quit and removes all watches
// associated with the inotify instance
func (w *Watcher) Close() error {
	if w.isClosed {
		return nil
	}
	w.isClosed = true

	// Remove all watches
	for name := range w.watches {
		w.Remove(name)
	}

	// Send "quit" message to the reader goroutine
	w.done <- true

	return nil
}

// Add starts watching on the named file.
func (w *Watcher) Add(name string) error {
	if w.isClosed {
		return errors.New("inotify instance already closed")
	}

	var flags uint32 = sys_AGNOSTIC_EVENTS

	w.mu.Lock()
	watchEntry, found := w.watches[name]
	w.mu.Unlock()
	if found {
		watchEntry.flags |= flags
		flags |= syscall.IN_MASK_ADD
	}
	wd, errno := syscall.InotifyAddWatch(w.fd, name, flags)
	if wd == -1 {
		return os.NewSyscallError("inotify_add_watch", errno)
	}

	w.mu.Lock()
	w.watches[name] = &watch{wd: uint32(wd), flags: flags}
	w.paths[wd] = name
	w.mu.Unlock()

	return nil
}

// Remove stops watching on the named file.
func (w *Watcher) Remove(name string) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	watch, ok := w.watches[name]
	if !ok {
		return errors.New(fmt.Sprintf("can't remove non-existent inotify watch for: %s", name))
	}
	success, errno := syscall.InotifyRmWatch(w.fd, watch.wd)
	if success == -1 {
		return os.NewSyscallError("inotify_rm_watch", errno)
	}
	delete(w.watches, name)
	return nil
}

// readEvents reads from the inotify file descriptor, converts the
// received events into Event objects and sends them via the Events channel
func (w *Watcher) readEvents() {
	var (
		buf   [syscall.SizeofInotifyEvent * 4096]byte // Buffer for a maximum of 4096 raw events
		n     int                                     // Number of bytes read with read()
		errno error                                   // Syscall errno
	)

	for {
		// See if there is a message on the "done" channel
		select {
		case <-w.done:
			syscall.Close(w.fd)
			close(w.Events)
			close(w.Errors)
			return
		default:
		}

		n, errno = syscall.Read(w.fd, buf[:])

		// If EOF is received
		if n == 0 {
			syscall.Close(w.fd)
			close(w.Events)
			close(w.Errors)
			return
		}

		if n < 0 {
			w.Errors <- os.NewSyscallError("read", errno)
			continue
		}
		if n < syscall.SizeofInotifyEvent {
			w.Errors <- errors.New("inotify: short read in readEvents()")
			continue
		}

		var offset uint32 = 0
		// We don't know how many events we just read into the buffer
		// While the offset points to at least one whole event...
		for offset <= uint32(n-syscall.SizeofInotifyEvent) {
			// Point "raw" to the event in the buffer
			raw := (*syscall.InotifyEvent)(unsafe.Pointer(&buf[offset]))

			mask := uint32(raw.Mask)
			nameLen := uint32(raw.Len)
			// If the event happened to the watched directory or the watched file, the kernel
			// doesn't append the filename to the event, but we would like to always fill the
			// the "Name" field with a valid filename. We retrieve the path of the watch from
			// the "paths" map.
			w.mu.Lock()
			name := w.paths[int(raw.Wd)]
			w.mu.Unlock()
			if nameLen > 0 {
				// Point "bytes" at the first byte of the filename
				bytes := (*[syscall.PathMax]byte)(unsafe.Pointer(&buf[offset+syscall.SizeofInotifyEvent]))
				// The filename is padded with NULL bytes. TrimRight() gets rid of those.
				name += "/" + strings.TrimRight(string(bytes[0:nameLen]), "\000")
			}

			event := newEvent(name, mask)

			// Send the events that are not ignored on the events channel
			if !event.ignoreLinux(mask) {
				w.Events <- event
			}

			// Move to the next event in the buffer
			offset += syscall.SizeofInotifyEvent + nameLen
		}
	}
}

// Certain types of events can be "ignored" and not sent over the Events
// channel. Such as events marked ignore by the kernel, or MODIFY events
// against files that do not exist.
func (e *Event) ignoreLinux(mask uint32) bool {
	// Ignore anything the inotify API says to ignore
	if mask&syscall.IN_IGNORED == syscall.IN_IGNORED {
		return true
	}

	// If the event is not a DELETE or RENAME, the file must exist.
	// Otherwise the event is ignored.
	// *Note*: this was put in place because it was seen that a MODIFY
	// event was sent after the DELETE. This ignores that MODIFY and
	// assumes a DELETE will come or has come if the file doesn't exist.
	if !(e.Op&Remove == Remove || e.Op&Rename == Rename) {
		_, statErr := os.Lstat(e.Name)
		return os.IsNotExist(statErr)
	}
	return false
}
