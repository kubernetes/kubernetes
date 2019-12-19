// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package inotify implements a wrapper for the Linux inotify system.

Example:
    watcher, err := inotify.NewWatcher()
    if err != nil {
        log.Fatal(err)
    }
    err = watcher.Watch("/tmp")
    if err != nil {
        log.Fatal(err)
    }
    for {
        select {
        case ev := <-watcher.Event:
            log.Println("event:", ev)
        case err := <-watcher.Error:
            log.Println("error:", err)
        }
    }

*/
package inotify // import "k8s.io/utils/inotify"

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"syscall"
	"unsafe"
)

// Event represents a notification
type Event struct {
	Mask   uint32 // Mask of events
	Cookie uint32 // Unique cookie associating related events (for rename(2))
	Name   string // File name (optional)
}

type watch struct {
	wd    uint32 // Watch descriptor (as returned by the inotify_add_watch() syscall)
	flags uint32 // inotify flags of this watch (see inotify(7) for the list of valid flags)
}

// Watcher represents an inotify instance
type Watcher struct {
	mu       sync.Mutex
	fd       int               // File descriptor (as returned by the inotify_init() syscall)
	watches  map[string]*watch // Map of inotify watches (key: path)
	paths    map[int]string    // Map of watched paths (key: watch descriptor)
	Error    chan error        // Errors are sent on this channel
	Event    chan *Event       // Events are returned on this channel
	done     chan bool         // Channel for sending a "quit message" to the reader goroutine
	isClosed bool              // Set to true when Close() is first called
}

// NewWatcher creates and returns a new inotify instance using inotify_init(2)
func NewWatcher() (*Watcher, error) {
	fd, errno := syscall.InotifyInit1(syscall.IN_CLOEXEC)
	if fd == -1 {
		return nil, os.NewSyscallError("inotify_init", errno)
	}
	w := &Watcher{
		fd:      fd,
		watches: make(map[string]*watch),
		paths:   make(map[int]string),
		Event:   make(chan *Event),
		Error:   make(chan error),
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

	// Send "quit" message to the reader goroutine
	w.done <- true
	for path := range w.watches {
		w.RemoveWatch(path)
	}

	return nil
}

// AddWatch adds path to the watched file set.
// The flags are interpreted as described in inotify_add_watch(2).
func (w *Watcher) AddWatch(path string, flags uint32) error {
	if w.isClosed {
		return errors.New("inotify instance already closed")
	}

	watchEntry, found := w.watches[path]
	if found {
		watchEntry.flags |= flags
		flags |= syscall.IN_MASK_ADD
	}

	w.mu.Lock() // synchronize with readEvents goroutine

	wd, err := syscall.InotifyAddWatch(w.fd, path, flags)
	if err != nil {
		w.mu.Unlock()
		return &os.PathError{
			Op:   "inotify_add_watch",
			Path: path,
			Err:  err,
		}
	}

	if !found {
		w.watches[path] = &watch{wd: uint32(wd), flags: flags}
		w.paths[wd] = path
	}
	w.mu.Unlock()
	return nil
}

// Watch adds path to the watched file set, watching all events.
func (w *Watcher) Watch(path string) error {
	return w.AddWatch(path, InAllEvents)
}

// RemoveWatch removes path from the watched file set.
func (w *Watcher) RemoveWatch(path string) error {
	watch, ok := w.watches[path]
	if !ok {
		return fmt.Errorf("can't remove non-existent inotify watch for: %s", path)
	}
	success, errno := syscall.InotifyRmWatch(w.fd, watch.wd)
	if success == -1 {
		return os.NewSyscallError("inotify_rm_watch", errno)
	}
	delete(w.watches, path)
	// Locking here to protect the read from paths in readEvents.
	w.mu.Lock()
	delete(w.paths, int(watch.wd))
	w.mu.Unlock()
	return nil
}

// readEvents reads from the inotify file descriptor, converts the
// received events into Event objects and sends them via the Event channel
func (w *Watcher) readEvents() {
	var buf [syscall.SizeofInotifyEvent * 4096]byte

	for {
		n, err := syscall.Read(w.fd, buf[:])
		// See if there is a message on the "done" channel
		var done bool
		select {
		case done = <-w.done:
		default:
		}

		// If EOF or a "done" message is received
		if n == 0 || done {
			// The syscall.Close can be slow.  Close
			// w.Event first.
			close(w.Event)
			err := syscall.Close(w.fd)
			if err != nil {
				w.Error <- os.NewSyscallError("close", err)
			}
			close(w.Error)
			return
		}
		if n < 0 {
			w.Error <- os.NewSyscallError("read", err)
			continue
		}
		if n < syscall.SizeofInotifyEvent {
			w.Error <- errors.New("inotify: short read in readEvents()")
			continue
		}

		var offset uint32
		// We don't know how many events we just read into the buffer
		// While the offset points to at least one whole event...
		for offset <= uint32(n-syscall.SizeofInotifyEvent) {
			// Point "raw" to the event in the buffer
			raw := (*syscall.InotifyEvent)(unsafe.Pointer(&buf[offset]))
			event := new(Event)
			event.Mask = uint32(raw.Mask)
			event.Cookie = uint32(raw.Cookie)
			nameLen := uint32(raw.Len)
			// If the event happened to the watched directory or the watched file, the kernel
			// doesn't append the filename to the event, but we would like to always fill the
			// the "Name" field with a valid filename. We retrieve the path of the watch from
			// the "paths" map.
			w.mu.Lock()
			name, ok := w.paths[int(raw.Wd)]
			w.mu.Unlock()
			if ok {
				event.Name = name
				if nameLen > 0 {
					// Point "bytes" at the first byte of the filename
					bytes := (*[syscall.PathMax]byte)(unsafe.Pointer(&buf[offset+syscall.SizeofInotifyEvent]))
					// The filename is padded with NUL bytes. TrimRight() gets rid of those.
					event.Name += "/" + strings.TrimRight(string(bytes[0:nameLen]), "\000")
				}
				// Send the event on the events channel
				w.Event <- event
			}
			// Move to the next event in the buffer
			offset += syscall.SizeofInotifyEvent + nameLen
		}
	}
}

// String formats the event e in the form
// "filename: 0xEventMask = IN_ACCESS|IN_ATTRIB_|..."
func (e *Event) String() string {
	var events string

	m := e.Mask
	for _, b := range eventBits {
		if m&b.Value == b.Value {
			m &^= b.Value
			events += "|" + b.Name
		}
	}

	if m != 0 {
		events += fmt.Sprintf("|%#x", m)
	}
	if len(events) > 0 {
		events = " == " + events[1:]
	}

	return fmt.Sprintf("%q: %#x%s", e.Name, e.Mask, events)
}

const (
	// Options for inotify_init() are not exported
	// IN_CLOEXEC    uint32 = syscall.IN_CLOEXEC
	// IN_NONBLOCK   uint32 = syscall.IN_NONBLOCK

	// Options for AddWatch

	// InDontFollow : Don't dereference pathname if it is a symbolic link
	InDontFollow uint32 = syscall.IN_DONT_FOLLOW
	// InOneshot : Monitor the filesystem object corresponding to pathname for one event, then remove from watch list
	InOneshot uint32 = syscall.IN_ONESHOT
	// InOnlydir : Watch pathname only if it is a directory
	InOnlydir uint32 = syscall.IN_ONLYDIR

	// The "IN_MASK_ADD" option is not exported, as AddWatch
	// adds it automatically, if there is already a watch for the given path
	// IN_MASK_ADD      uint32 = syscall.IN_MASK_ADD

	// Events

	// InAccess : File was accessed
	InAccess uint32 = syscall.IN_ACCESS
	// InAllEvents : Bit mask for all notify events
	InAllEvents uint32 = syscall.IN_ALL_EVENTS
	// InAttrib : Metadata changed
	InAttrib uint32 = syscall.IN_ATTRIB
	// InClose : Equates to IN_CLOSE_WRITE | IN_CLOSE_NOWRITE
	InClose uint32 = syscall.IN_CLOSE
	// InCloseNowrite : File or directory not opened for writing was closed
	InCloseNowrite uint32 = syscall.IN_CLOSE_NOWRITE
	// InCloseWrite : File opened for writing was closed
	InCloseWrite uint32 = syscall.IN_CLOSE_WRITE
	// InCreate : File/directory created in watched directory
	InCreate uint32 = syscall.IN_CREATE
	// InDelete : File/directory deleted from watched directory
	InDelete uint32 = syscall.IN_DELETE
	// InDeleteSelf : Watched file/directory was itself deleted
	InDeleteSelf uint32 = syscall.IN_DELETE_SELF
	// InModify : File was modified
	InModify uint32 = syscall.IN_MODIFY
	// InMove : Equates to IN_MOVED_FROM | IN_MOVED_TO
	InMove uint32 = syscall.IN_MOVE
	// InMovedFrom : Generated for the directory containing the old filename when a file is renamed
	InMovedFrom uint32 = syscall.IN_MOVED_FROM
	// InMovedTo : Generated for the directory containing the new filename when a file is renamed
	InMovedTo uint32 = syscall.IN_MOVED_TO
	// InMoveSelf : Watched file/directory was itself moved
	InMoveSelf uint32 = syscall.IN_MOVE_SELF
	// InOpen : File or directory was opened
	InOpen uint32 = syscall.IN_OPEN

	// Special events

	// InIsdir : Subject of this event is a directory
	InIsdir uint32 = syscall.IN_ISDIR
	// InIgnored : Watch was removed explicitly or automatically
	InIgnored uint32 = syscall.IN_IGNORED
	// InQOverflow : Event queue overflowed
	InQOverflow uint32 = syscall.IN_Q_OVERFLOW
	// InUnmount : Filesystem containing watched object was unmounted
	InUnmount uint32 = syscall.IN_UNMOUNT
)

var eventBits = []struct {
	Value uint32
	Name  string
}{
	{InAccess, "IN_ACCESS"},
	{InAttrib, "IN_ATTRIB"},
	{InClose, "IN_CLOSE"},
	{InCloseNowrite, "IN_CLOSE_NOWRITE"},
	{InCloseWrite, "IN_CLOSE_WRITE"},
	{InCreate, "IN_CREATE"},
	{InDelete, "IN_DELETE"},
	{InDeleteSelf, "IN_DELETE_SELF"},
	{InModify, "IN_MODIFY"},
	{InMove, "IN_MOVE"},
	{InMovedFrom, "IN_MOVED_FROM"},
	{InMovedTo, "IN_MOVED_TO"},
	{InMoveSelf, "IN_MOVE_SELF"},
	{InOpen, "IN_OPEN"},
	{InIsdir, "IN_ISDIR"},
	{InIgnored, "IN_IGNORED"},
	{InQOverflow, "IN_Q_OVERFLOW"},
	{InUnmount, "IN_UNMOUNT"},
}
