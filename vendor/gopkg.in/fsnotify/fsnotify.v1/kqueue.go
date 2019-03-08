// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd openbsd netbsd dragonfly darwin

package fsnotify

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"time"
)

// Watcher watches a set of files, delivering events to a channel.
type Watcher struct {
	Events chan Event
	Errors chan error
	done   chan bool // Channel for sending a "quit message" to the reader goroutine

	kq int // File descriptor (as returned by the kqueue() syscall).

	mu              sync.Mutex        // Protects access to watcher data
	watches         map[string]int    // Map of watched file descriptors (key: path).
	externalWatches map[string]bool   // Map of watches added by user of the library.
	dirFlags        map[string]uint32 // Map of watched directories to fflags used in kqueue.
	paths           map[int]pathInfo  // Map file descriptors to path names for processing kqueue events.
	fileExists      map[string]bool   // Keep track of if we know this file exists (to stop duplicate create events).
	isClosed        bool              // Set to true when Close() is first called
}

type pathInfo struct {
	name  string
	isDir bool
}

// NewWatcher establishes a new watcher with the underlying OS and begins waiting for events.
func NewWatcher() (*Watcher, error) {
	kq, err := kqueue()
	if err != nil {
		return nil, err
	}

	w := &Watcher{
		kq:              kq,
		watches:         make(map[string]int),
		dirFlags:        make(map[string]uint32),
		paths:           make(map[int]pathInfo),
		fileExists:      make(map[string]bool),
		externalWatches: make(map[string]bool),
		Events:          make(chan Event),
		Errors:          make(chan error),
		done:            make(chan bool),
	}

	go w.readEvents()
	return w, nil
}

// Close removes all watches and closes the events channel.
func (w *Watcher) Close() error {
	w.mu.Lock()
	if w.isClosed {
		w.mu.Unlock()
		return nil
	}
	w.isClosed = true
	w.mu.Unlock()

	w.mu.Lock()
	ws := w.watches
	w.mu.Unlock()

	var err error
	for name := range ws {
		if e := w.Remove(name); e != nil && err == nil {
			err = e
		}
	}

	// Send "quit" message to the reader goroutine:
	w.done <- true

	return nil
}

// Add starts watching the named file or directory (non-recursively).
func (w *Watcher) Add(name string) error {
	w.mu.Lock()
	w.externalWatches[name] = true
	w.mu.Unlock()
	return w.addWatch(name, noteAllEvents)
}

// Remove stops watching the the named file or directory (non-recursively).
func (w *Watcher) Remove(name string) error {
	name = filepath.Clean(name)
	w.mu.Lock()
	watchfd, ok := w.watches[name]
	w.mu.Unlock()
	if !ok {
		return fmt.Errorf("can't remove non-existent kevent watch for: %s", name)
	}

	const registerRemove = syscall.EV_DELETE
	if err := register(w.kq, []int{watchfd}, registerRemove, 0); err != nil {
		return err
	}

	syscall.Close(watchfd)

	w.mu.Lock()
	isDir := w.paths[watchfd].isDir
	delete(w.watches, name)
	delete(w.paths, watchfd)
	delete(w.dirFlags, name)
	w.mu.Unlock()

	// Find all watched paths that are in this directory that are not external.
	if isDir {
		var pathsToRemove []string
		w.mu.Lock()
		for _, path := range w.paths {
			wdir, _ := filepath.Split(path.name)
			if filepath.Clean(wdir) == name {
				if !w.externalWatches[path.name] {
					pathsToRemove = append(pathsToRemove, path.name)
				}
			}
		}
		w.mu.Unlock()
		for _, name := range pathsToRemove {
			// Since these are internal, not much sense in propagating error
			// to the user, as that will just confuse them with an error about
			// a path they did not explicitly watch themselves.
			w.Remove(name)
		}
	}

	return nil
}

// Watch all events (except NOTE_EXTEND, NOTE_LINK, NOTE_REVOKE)
const noteAllEvents = syscall.NOTE_DELETE | syscall.NOTE_WRITE | syscall.NOTE_ATTRIB | syscall.NOTE_RENAME

// keventWaitTime to block on each read from kevent
var keventWaitTime = durationToTimespec(100 * time.Millisecond)

// addWatch adds name to the watched file set.
// The flags are interpreted as described in kevent(2).
func (w *Watcher) addWatch(name string, flags uint32) error {
	var isDir bool
	// Make ./name and name equivalent
	name = filepath.Clean(name)

	w.mu.Lock()
	if w.isClosed {
		w.mu.Unlock()
		return errors.New("kevent instance already closed")
	}
	watchfd, alreadyWatching := w.watches[name]
	// We already have a watch, but we can still override flags.
	if alreadyWatching {
		isDir = w.paths[watchfd].isDir
	}
	w.mu.Unlock()

	if !alreadyWatching {
		fi, err := os.Lstat(name)
		if err != nil {
			return err
		}

		// Don't watch sockets.
		if fi.Mode()&os.ModeSocket == os.ModeSocket {
			return nil
		}

		// Don't watch named pipes.
		if fi.Mode()&os.ModeNamedPipe == os.ModeNamedPipe {
			return nil
		}

		// Follow Symlinks
		// Unfortunately, Linux can add bogus symlinks to watch list without
		// issue, and Windows can't do symlinks period (AFAIK). To  maintain
		// consistency, we will act like everything is fine. There will simply
		// be no file events for broken symlinks.
		// Hence the returns of nil on errors.
		if fi.Mode()&os.ModeSymlink == os.ModeSymlink {
			name, err = filepath.EvalSymlinks(name)
			if err != nil {
				return nil
			}

			fi, err = os.Lstat(name)
			if err != nil {
				return nil
			}
		}

		watchfd, err = syscall.Open(name, openMode, 0700)
		if watchfd == -1 {
			return err
		}

		isDir = fi.IsDir()
	}

	const registerAdd = syscall.EV_ADD | syscall.EV_CLEAR | syscall.EV_ENABLE
	if err := register(w.kq, []int{watchfd}, registerAdd, flags); err != nil {
		syscall.Close(watchfd)
		return err
	}

	if !alreadyWatching {
		w.mu.Lock()
		w.watches[name] = watchfd
		w.paths[watchfd] = pathInfo{name: name, isDir: isDir}
		w.mu.Unlock()
	}

	if isDir {
		// Watch the directory if it has not been watched before,
		// or if it was watched before, but perhaps only a NOTE_DELETE (watchDirectoryFiles)
		w.mu.Lock()
		watchDir := (flags&syscall.NOTE_WRITE) == syscall.NOTE_WRITE &&
			(!alreadyWatching || (w.dirFlags[name]&syscall.NOTE_WRITE) != syscall.NOTE_WRITE)
		// Store flags so this watch can be updated later
		w.dirFlags[name] = flags
		w.mu.Unlock()

		if watchDir {
			if err := w.watchDirectoryFiles(name); err != nil {
				return err
			}
		}
	}
	return nil
}

// readEvents reads from kqueue and converts the received kevents into
// Event values that it sends down the Events channel.
func (w *Watcher) readEvents() {
	eventBuffer := make([]syscall.Kevent_t, 10)

	for {
		// See if there is a message on the "done" channel
		select {
		case <-w.done:
			err := syscall.Close(w.kq)
			if err != nil {
				w.Errors <- err
			}
			close(w.Events)
			close(w.Errors)
			return
		default:
		}

		// Get new events
		kevents, err := read(w.kq, eventBuffer, &keventWaitTime)
		// EINTR is okay, the syscall was interrupted before timeout expired.
		if err != nil && err != syscall.EINTR {
			w.Errors <- err
			continue
		}

		// Flush the events we received to the Events channel
		for len(kevents) > 0 {
			kevent := &kevents[0]
			watchfd := int(kevent.Ident)
			mask := uint32(kevent.Fflags)
			w.mu.Lock()
			path := w.paths[watchfd]
			w.mu.Unlock()
			event := newEvent(path.name, mask)

			if path.isDir && !(event.Op&Remove == Remove) {
				// Double check to make sure the directory exists. This can happen when
				// we do a rm -fr on a recursively watched folders and we receive a
				// modification event first but the folder has been deleted and later
				// receive the delete event
				if _, err := os.Lstat(event.Name); os.IsNotExist(err) {
					// mark is as delete event
					event.Op |= Remove
				}
			}

			if event.Op&Rename == Rename || event.Op&Remove == Remove {
				w.Remove(event.Name)
				w.mu.Lock()
				delete(w.fileExists, event.Name)
				w.mu.Unlock()
			}

			if path.isDir && event.Op&Write == Write && !(event.Op&Remove == Remove) {
				w.sendDirectoryChangeEvents(event.Name)
			} else {
				// Send the event on the Events channel
				w.Events <- event
			}

			if event.Op&Remove == Remove {
				// Look for a file that may have overwritten this.
				// For example, mv f1 f2 will delete f2, then create f2.
				fileDir, _ := filepath.Split(event.Name)
				fileDir = filepath.Clean(fileDir)
				w.mu.Lock()
				_, found := w.watches[fileDir]
				w.mu.Unlock()
				if found {
					// make sure the directory exists before we watch for changes. When we
					// do a recursive watch and perform rm -fr, the parent directory might
					// have gone missing, ignore the missing directory and let the
					// upcoming delete event remove the watch from the parent directory.
					if _, err := os.Lstat(fileDir); os.IsExist(err) {
						w.sendDirectoryChangeEvents(fileDir)
						// FIXME: should this be for events on files or just isDir?
					}
				}
			}

			// Move to next event
			kevents = kevents[1:]
		}
	}
}

// newEvent returns an platform-independent Event based on kqueue Fflags.
func newEvent(name string, mask uint32) Event {
	e := Event{Name: name}
	if mask&syscall.NOTE_DELETE == syscall.NOTE_DELETE {
		e.Op |= Remove
	}
	if mask&syscall.NOTE_WRITE == syscall.NOTE_WRITE {
		e.Op |= Write
	}
	if mask&syscall.NOTE_RENAME == syscall.NOTE_RENAME {
		e.Op |= Rename
	}
	if mask&syscall.NOTE_ATTRIB == syscall.NOTE_ATTRIB {
		e.Op |= Chmod
	}
	return e
}

func newCreateEvent(name string) Event {
	return Event{Name: name, Op: Create}
}

// watchDirectoryFiles to mimic inotify when adding a watch on a directory
func (w *Watcher) watchDirectoryFiles(dirPath string) error {
	// Get all files
	files, err := ioutil.ReadDir(dirPath)
	if err != nil {
		return err
	}

	for _, fileInfo := range files {
		filePath := filepath.Join(dirPath, fileInfo.Name())
		if err := w.internalWatch(filePath, fileInfo); err != nil {
			return err
		}

		w.mu.Lock()
		w.fileExists[filePath] = true
		w.mu.Unlock()
	}

	return nil
}

// sendDirectoryEvents searches the directory for newly created files
// and sends them over the event channel. This functionality is to have
// the BSD version of fsnotify match Linux inotify which provides a
// create event for files created in a watched directory.
func (w *Watcher) sendDirectoryChangeEvents(dirPath string) {
	// Get all files
	files, err := ioutil.ReadDir(dirPath)
	if err != nil {
		w.Errors <- err
	}

	// Search for new files
	for _, fileInfo := range files {
		filePath := filepath.Join(dirPath, fileInfo.Name())
		w.mu.Lock()
		_, doesExist := w.fileExists[filePath]
		w.mu.Unlock()
		if !doesExist {
			// Send create event
			w.Events <- newCreateEvent(filePath)
		}

		// like watchDirectoryFiles (but without doing another ReadDir)
		if err := w.internalWatch(filePath, fileInfo); err != nil {
			return
		}

		w.mu.Lock()
		w.fileExists[filePath] = true
		w.mu.Unlock()
	}
}

func (w *Watcher) internalWatch(name string, fileInfo os.FileInfo) error {
	if fileInfo.IsDir() {
		// mimic Linux providing delete events for subdirectories
		// but preserve the flags used if currently watching subdirectory
		w.mu.Lock()
		flags := w.dirFlags[name]
		w.mu.Unlock()

		flags |= syscall.NOTE_DELETE
		return w.addWatch(name, flags)
	}

	// watch file to mimic Linux inotify
	return w.addWatch(name, noteAllEvents)
}

// kqueue creates a new kernel event queue and returns a descriptor.
func kqueue() (kq int, err error) {
	kq, err = syscall.Kqueue()
	if kq == -1 {
		return kq, err
	}
	return kq, nil
}

// register events with the queue
func register(kq int, fds []int, flags int, fflags uint32) error {
	changes := make([]syscall.Kevent_t, len(fds))

	for i, fd := range fds {
		// SetKevent converts int to the platform-specific types:
		syscall.SetKevent(&changes[i], fd, syscall.EVFILT_VNODE, flags)
		changes[i].Fflags = fflags
	}

	// register the events
	success, err := syscall.Kevent(kq, changes, nil, nil)
	if success == -1 {
		return err
	}
	return nil
}

// read retrieves pending events, or waits until an event occurs.
// A timeout of nil blocks indefinitely, while 0 polls the queue.
func read(kq int, events []syscall.Kevent_t, timeout *syscall.Timespec) ([]syscall.Kevent_t, error) {
	n, err := syscall.Kevent(kq, nil, events, timeout)
	if err != nil {
		return nil, err
	}
	return events[0:n], nil
}

// durationToTimespec prepares a timeout value
func durationToTimespec(d time.Duration) syscall.Timespec {
	return syscall.NsecToTimespec(d.Nanoseconds())
}
