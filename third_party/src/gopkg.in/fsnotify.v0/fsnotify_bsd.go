// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd openbsd netbsd darwin

package fsnotify

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"syscall"
)

const (
	// Watch all events
	sys_NOTE_ALLEVENTS = syscall.NOTE_DELETE | syscall.NOTE_WRITE | syscall.NOTE_ATTRIB | syscall.NOTE_RENAME

	// Block for 100 ms on each call to kevent
	keventWaitTime = 100e6
)

func newEvent(name string, mask uint32, create bool) Event {
	e := Event{Name: name}
	if create {
		e.Op |= Create
	}
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

type Watcher struct {
	mu              sync.Mutex          // Mutex for the Watcher itself.
	kq              int                 // File descriptor (as returned by the kqueue() syscall)
	watches         map[string]int      // Map of watched file descriptors (key: path)
	wmut            sync.Mutex          // Protects access to watches.
	enFlags         map[string]uint32   // Map of watched files to evfilt note flags used in kqueue
	enmut           sync.Mutex          // Protects access to enFlags.
	paths           map[int]string      // Map of watched paths (key: watch descriptor)
	finfo           map[int]os.FileInfo // Map of file information (isDir, isReg; key: watch descriptor)
	pmut            sync.Mutex          // Protects access to paths and finfo.
	fileExists      map[string]bool     // Keep track of if we know this file exists (to stop duplicate create events)
	femut           sync.Mutex          // Protects access to fileExists.
	externalWatches map[string]bool     // Map of watches added by user of the library.
	ewmut           sync.Mutex          // Protects access to externalWatches.
	Errors          chan error          // Errors are sent on this channel
	Events          chan Event          // Events are returned on this channel
	done            chan bool           // Channel for sending a "quit message" to the reader goroutine
	isClosed        bool                // Set to true when Close() is first called
}

// NewWatcher creates and returns a new kevent instance using kqueue(2)
func NewWatcher() (*Watcher, error) {
	fd, errno := syscall.Kqueue()
	if fd == -1 {
		return nil, os.NewSyscallError("kqueue", errno)
	}
	w := &Watcher{
		kq:              fd,
		watches:         make(map[string]int),
		enFlags:         make(map[string]uint32),
		paths:           make(map[int]string),
		finfo:           make(map[int]os.FileInfo),
		fileExists:      make(map[string]bool),
		externalWatches: make(map[string]bool),
		Events:          make(chan Event),
		Errors:          make(chan error),
		done:            make(chan bool, 1),
	}

	go w.readEvents()
	return w, nil
}

// Close closes a kevent watcher instance
// It sends a message to the reader goroutine to quit and removes all watches
// associated with the kevent instance
func (w *Watcher) Close() error {
	w.mu.Lock()
	if w.isClosed {
		w.mu.Unlock()
		return nil
	}
	w.isClosed = true
	w.mu.Unlock()

	// Send "quit" message to the reader goroutine
	w.done <- true
	w.wmut.Lock()
	ws := w.watches
	w.wmut.Unlock()
	for name := range ws {
		w.Remove(name)
	}

	return nil
}

// AddWatch adds path to the watched file set.
// The flags are interpreted as described in kevent(2).
func (w *Watcher) addWatch(path string, flags uint32) error {
	w.mu.Lock()
	if w.isClosed {
		w.mu.Unlock()
		return errors.New("kevent instance already closed")
	}
	w.mu.Unlock()

	watchDir := false

	w.wmut.Lock()
	watchfd, found := w.watches[path]
	w.wmut.Unlock()
	if !found {
		fi, errstat := os.Lstat(path)
		if errstat != nil {
			return errstat
		}

		// don't watch socket
		if fi.Mode()&os.ModeSocket == os.ModeSocket {
			return nil
		}

		// Follow Symlinks
		// Unfortunately, Linux can add bogus symlinks to watch list without
		// issue, and Windows can't do symlinks period (AFAIK). To  maintain
		// consistency, we will act like everything is fine. There will simply
		// be no file events for broken symlinks.
		// Hence the returns of nil on errors.
		if fi.Mode()&os.ModeSymlink == os.ModeSymlink {
			path, err := filepath.EvalSymlinks(path)
			if err != nil {
				return nil
			}

			fi, errstat = os.Lstat(path)
			if errstat != nil {
				return nil
			}
		}

		fd, errno := syscall.Open(path, open_FLAGS, 0700)
		if fd == -1 {
			return os.NewSyscallError("Open", errno)
		}
		watchfd = fd

		w.wmut.Lock()
		w.watches[path] = watchfd
		w.wmut.Unlock()

		w.pmut.Lock()
		w.paths[watchfd] = path
		w.finfo[watchfd] = fi
		w.pmut.Unlock()
	}
	// Watch the directory if it has not been watched before.
	w.pmut.Lock()
	w.enmut.Lock()
	if w.finfo[watchfd].IsDir() &&
		(flags&syscall.NOTE_WRITE) == syscall.NOTE_WRITE &&
		(!found || (w.enFlags[path]&syscall.NOTE_WRITE) != syscall.NOTE_WRITE) {
		watchDir = true
	}
	w.enmut.Unlock()
	w.pmut.Unlock()

	w.enmut.Lock()
	w.enFlags[path] = flags
	w.enmut.Unlock()

	var kbuf [1]syscall.Kevent_t
	watchEntry := &kbuf[0]
	watchEntry.Fflags = flags
	syscall.SetKevent(watchEntry, watchfd, syscall.EVFILT_VNODE, syscall.EV_ADD|syscall.EV_CLEAR)
	entryFlags := watchEntry.Flags
	success, errno := syscall.Kevent(w.kq, kbuf[:], nil, nil)
	if success == -1 {
		return errno
	} else if (entryFlags & syscall.EV_ERROR) == syscall.EV_ERROR {
		return errors.New("kevent add error")
	}

	if watchDir {
		errdir := w.watchDirectoryFiles(path)
		if errdir != nil {
			return errdir
		}
	}
	return nil
}

// Add starts watching on the named file.
func (w *Watcher) Add(name string) error {
	w.ewmut.Lock()
	w.externalWatches[name] = true
	w.ewmut.Unlock()
	return w.addWatch(name, sys_NOTE_ALLEVENTS)
}

// Remove stops watching on the named file.
func (w *Watcher) Remove(name string) error {
	w.wmut.Lock()
	watchfd, ok := w.watches[name]
	w.wmut.Unlock()
	if !ok {
		return errors.New(fmt.Sprintf("can't remove non-existent kevent watch for: %s", name))
	}
	var kbuf [1]syscall.Kevent_t
	watchEntry := &kbuf[0]
	syscall.SetKevent(watchEntry, watchfd, syscall.EVFILT_VNODE, syscall.EV_DELETE)
	entryFlags := watchEntry.Flags
	success, errno := syscall.Kevent(w.kq, kbuf[:], nil, nil)
	if success == -1 {
		return os.NewSyscallError("kevent_rm_watch", errno)
	} else if (entryFlags & syscall.EV_ERROR) == syscall.EV_ERROR {
		return errors.New("kevent rm error")
	}
	syscall.Close(watchfd)
	w.wmut.Lock()
	delete(w.watches, name)
	w.wmut.Unlock()
	w.enmut.Lock()
	delete(w.enFlags, name)
	w.enmut.Unlock()
	w.pmut.Lock()
	delete(w.paths, watchfd)
	fInfo := w.finfo[watchfd]
	delete(w.finfo, watchfd)
	w.pmut.Unlock()

	// Find all watched paths that are in this directory that are not external.
	if fInfo.IsDir() {
		var pathsToRemove []string
		w.pmut.Lock()
		for _, wpath := range w.paths {
			wdir, _ := filepath.Split(wpath)
			if filepath.Clean(wdir) == filepath.Clean(name) {
				w.ewmut.Lock()
				if !w.externalWatches[wpath] {
					pathsToRemove = append(pathsToRemove, wpath)
				}
				w.ewmut.Unlock()
			}
		}
		w.pmut.Unlock()
		for _, name := range pathsToRemove {
			// Since these are internal, not much sense in propagating error
			// to the user, as that will just confuse them with an error about
			// a path they did not explicitly watch themselves.
			w.Remove(name)
		}
	}

	return nil
}

// readEvents reads from the kqueue file descriptor, converts the
// received events into Event objects and sends them via the Events channel
func (w *Watcher) readEvents() {
	var (
		keventbuf [10]syscall.Kevent_t // Event buffer
		kevents   []syscall.Kevent_t   // Received events
		twait     *syscall.Timespec    // Time to block waiting for events
		n         int                  // Number of events returned from kevent
		errno     error                // Syscall errno
	)
	kevents = keventbuf[0:0]
	twait = new(syscall.Timespec)
	*twait = syscall.NsecToTimespec(keventWaitTime)

	for {
		// See if there is a message on the "done" channel
		var done bool
		select {
		case done = <-w.done:
		default:
		}

		// If "done" message is received
		if done {
			errno := syscall.Close(w.kq)
			if errno != nil {
				w.Errors <- os.NewSyscallError("close", errno)
			}
			close(w.Events)
			close(w.Errors)
			return
		}

		// Get new events
		if len(kevents) == 0 {
			n, errno = syscall.Kevent(w.kq, nil, keventbuf[:], twait)

			// EINTR is okay, basically the syscall was interrupted before
			// timeout expired.
			if errno != nil && errno != syscall.EINTR {
				w.Errors <- os.NewSyscallError("kevent", errno)
				continue
			}

			// Received some events
			if n > 0 {
				kevents = keventbuf[0:n]
			}
		}

		// Flush the events we received to the Events channel
		for len(kevents) > 0 {
			watchEvent := &kevents[0]
			mask := uint32(watchEvent.Fflags)
			w.pmut.Lock()
			name := w.paths[int(watchEvent.Ident)]
			fileInfo := w.finfo[int(watchEvent.Ident)]
			w.pmut.Unlock()

			event := newEvent(name, mask, false)

			if fileInfo != nil && fileInfo.IsDir() && !(event.Op&Remove == Remove) {
				// Double check to make sure the directory exist. This can happen when
				// we do a rm -fr on a recursively watched folders and we receive a
				// modification event first but the folder has been deleted and later
				// receive the delete event
				if _, err := os.Lstat(event.Name); os.IsNotExist(err) {
					// mark is as delete event
					event.Op |= Remove
				}
			}

			if fileInfo != nil && fileInfo.IsDir() && event.Op&Write == Write && !(event.Op&Remove == Remove) {
				w.sendDirectoryChangeEvents(event.Name)
			} else {
				// Send the event on the Events channel
				w.Events <- event
			}

			// Move to next event
			kevents = kevents[1:]

			if event.Op&Rename == Rename {
				w.Remove(event.Name)
				w.femut.Lock()
				delete(w.fileExists, event.Name)
				w.femut.Unlock()
			}
			if event.Op&Remove == Remove {
				w.Remove(event.Name)
				w.femut.Lock()
				delete(w.fileExists, event.Name)
				w.femut.Unlock()

				// Look for a file that may have overwritten this
				// (ie mv f1 f2 will delete f2 then create f2)
				fileDir, _ := filepath.Split(event.Name)
				fileDir = filepath.Clean(fileDir)
				w.wmut.Lock()
				_, found := w.watches[fileDir]
				w.wmut.Unlock()
				if found {
					// make sure the directory exist before we watch for changes. When we
					// do a recursive watch and perform rm -fr, the parent directory might
					// have gone missing, ignore the missing directory and let the
					// upcoming delete event remove the watch form the parent folder
					if _, err := os.Lstat(fileDir); !os.IsNotExist(err) {
						w.sendDirectoryChangeEvents(fileDir)
					}
				}
			}
		}
	}
}

func (w *Watcher) watchDirectoryFiles(dirPath string) error {
	// Get all files
	files, err := ioutil.ReadDir(dirPath)
	if err != nil {
		return err
	}

	// Search for new files
	for _, fileInfo := range files {
		filePath := filepath.Join(dirPath, fileInfo.Name())

		if fileInfo.IsDir() == false {
			// Watch file to mimic linux fsnotify
			e := w.addWatch(filePath, sys_NOTE_ALLEVENTS)
			if e != nil {
				return e
			}
		} else {
			// If the user is currently watching directory
			// we want to preserve the flags used
			w.enmut.Lock()
			currFlags, found := w.enFlags[filePath]
			w.enmut.Unlock()
			var newFlags uint32 = syscall.NOTE_DELETE
			if found {
				newFlags |= currFlags
			}

			// Linux gives deletes if not explicitly watching
			e := w.addWatch(filePath, newFlags)
			if e != nil {
				return e
			}
		}
		w.femut.Lock()
		w.fileExists[filePath] = true
		w.femut.Unlock()
	}

	return nil
}

// sendDirectoryEvents searches the directory for newly created files
// and sends them over the event channel. This functionality is to have
// the BSD version of fsnotify match linux fsnotify which provides a
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
		w.femut.Lock()
		_, doesExist := w.fileExists[filePath]
		w.femut.Unlock()
		if !doesExist {
			// Send create event (mask=0)
			event := newEvent(filePath, 0, true)
			w.Events <- event
		}
		w.femut.Lock()
		w.fileExists[filePath] = true
		w.femut.Unlock()
	}
	w.watchDirectoryFiles(dirPath)
}
