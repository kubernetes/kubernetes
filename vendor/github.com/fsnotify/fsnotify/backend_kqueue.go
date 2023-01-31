//go:build freebsd || openbsd || netbsd || dragonfly || darwin
// +build freebsd openbsd netbsd dragonfly darwin

package fsnotify

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"

	"golang.org/x/sys/unix"
)

// Watcher watches a set of paths, delivering events on a channel.
//
// A watcher should not be copied (e.g. pass it by pointer, rather than by
// value).
//
// # Linux notes
//
// When a file is removed a Remove event won't be emitted until all file
// descriptors are closed, and deletes will always emit a Chmod. For example:
//
//     fp := os.Open("file")
//     os.Remove("file")        // Triggers Chmod
//     fp.Close()               // Triggers Remove
//
// This is the event that inotify sends, so not much can be changed about this.
//
// The fs.inotify.max_user_watches sysctl variable specifies the upper limit
// for the number of watches per user, and fs.inotify.max_user_instances
// specifies the maximum number of inotify instances per user. Every Watcher you
// create is an "instance", and every path you add is a "watch".
//
// These are also exposed in /proc as /proc/sys/fs/inotify/max_user_watches and
// /proc/sys/fs/inotify/max_user_instances
//
// To increase them you can use sysctl or write the value to the /proc file:
//
//     # Default values on Linux 5.18
//     sysctl fs.inotify.max_user_watches=124983
//     sysctl fs.inotify.max_user_instances=128
//
// To make the changes persist on reboot edit /etc/sysctl.conf or
// /usr/lib/sysctl.d/50-default.conf (details differ per Linux distro; check
// your distro's documentation):
//
//     fs.inotify.max_user_watches=124983
//     fs.inotify.max_user_instances=128
//
// Reaching the limit will result in a "no space left on device" or "too many open
// files" error.
//
// # kqueue notes (macOS, BSD)
//
// kqueue requires opening a file descriptor for every file that's being watched;
// so if you're watching a directory with five files then that's six file
// descriptors. You will run in to your system's "max open files" limit faster on
// these platforms.
//
// The sysctl variables kern.maxfiles and kern.maxfilesperproc can be used to
// control the maximum number of open files, as well as /etc/login.conf on BSD
// systems.
//
// # macOS notes
//
// Spotlight indexing on macOS can result in multiple events (see [#15]). A
// temporary workaround is to add your folder(s) to the "Spotlight Privacy
// Settings" until we have a native FSEvents implementation (see [#11]).
//
// [#11]: https://github.com/fsnotify/fsnotify/issues/11
// [#15]: https://github.com/fsnotify/fsnotify/issues/15
type Watcher struct {
	// Events sends the filesystem change events.
	//
	// fsnotify can send the following events; a "path" here can refer to a
	// file, directory, symbolic link, or special file like a FIFO.
	//
	//   fsnotify.Create    A new path was created; this may be followed by one
	//                      or more Write events if data also gets written to a
	//                      file.
	//
	//   fsnotify.Remove    A path was removed.
	//
	//   fsnotify.Rename    A path was renamed. A rename is always sent with the
	//                      old path as Event.Name, and a Create event will be
	//                      sent with the new name. Renames are only sent for
	//                      paths that are currently watched; e.g. moving an
	//                      unmonitored file into a monitored directory will
	//                      show up as just a Create. Similarly, renaming a file
	//                      to outside a monitored directory will show up as
	//                      only a Rename.
	//
	//   fsnotify.Write     A file or named pipe was written to. A Truncate will
	//                      also trigger a Write. A single "write action"
	//                      initiated by the user may show up as one or multiple
	//                      writes, depending on when the system syncs things to
	//                      disk. For example when compiling a large Go program
	//                      you may get hundreds of Write events, so you
	//                      probably want to wait until you've stopped receiving
	//                      them (see the dedup example in cmd/fsnotify).
	//
	//   fsnotify.Chmod     Attributes were changed. On Linux this is also sent
	//                      when a file is removed (or more accurately, when a
	//                      link to an inode is removed). On kqueue it's sent
	//                      and on kqueue when a file is truncated. On Windows
	//                      it's never sent.
	Events chan Event

	// Errors sends any errors.
	Errors chan error

	done         chan struct{}
	kq           int                         // File descriptor (as returned by the kqueue() syscall).
	closepipe    [2]int                      // Pipe used for closing.
	mu           sync.Mutex                  // Protects access to watcher data
	watches      map[string]int              // Watched file descriptors (key: path).
	watchesByDir map[string]map[int]struct{} // Watched file descriptors indexed by the parent directory (key: dirname(path)).
	userWatches  map[string]struct{}         // Watches added with Watcher.Add()
	dirFlags     map[string]uint32           // Watched directories to fflags used in kqueue.
	paths        map[int]pathInfo            // File descriptors to path names for processing kqueue events.
	fileExists   map[string]struct{}         // Keep track of if we know this file exists (to stop duplicate create events).
	isClosed     bool                        // Set to true when Close() is first called
}

type pathInfo struct {
	name  string
	isDir bool
}

// NewWatcher creates a new Watcher.
func NewWatcher() (*Watcher, error) {
	kq, closepipe, err := newKqueue()
	if err != nil {
		return nil, err
	}

	w := &Watcher{
		kq:           kq,
		closepipe:    closepipe,
		watches:      make(map[string]int),
		watchesByDir: make(map[string]map[int]struct{}),
		dirFlags:     make(map[string]uint32),
		paths:        make(map[int]pathInfo),
		fileExists:   make(map[string]struct{}),
		userWatches:  make(map[string]struct{}),
		Events:       make(chan Event),
		Errors:       make(chan error),
		done:         make(chan struct{}),
	}

	go w.readEvents()
	return w, nil
}

// newKqueue creates a new kernel event queue and returns a descriptor.
//
// This registers a new event on closepipe, which will trigger an event when
// it's closed. This way we can use kevent() without timeout/polling; without
// the closepipe, it would block forever and we wouldn't be able to stop it at
// all.
func newKqueue() (kq int, closepipe [2]int, err error) {
	kq, err = unix.Kqueue()
	if kq == -1 {
		return kq, closepipe, err
	}

	// Register the close pipe.
	err = unix.Pipe(closepipe[:])
	if err != nil {
		unix.Close(kq)
		return kq, closepipe, err
	}

	// Register changes to listen on the closepipe.
	changes := make([]unix.Kevent_t, 1)
	// SetKevent converts int to the platform-specific types.
	unix.SetKevent(&changes[0], closepipe[0], unix.EVFILT_READ,
		unix.EV_ADD|unix.EV_ENABLE|unix.EV_ONESHOT)

	ok, err := unix.Kevent(kq, changes, nil, nil)
	if ok == -1 {
		unix.Close(kq)
		unix.Close(closepipe[0])
		unix.Close(closepipe[1])
		return kq, closepipe, err
	}
	return kq, closepipe, nil
}

// Returns true if the event was sent, or false if watcher is closed.
func (w *Watcher) sendEvent(e Event) bool {
	select {
	case w.Events <- e:
		return true
	case <-w.done:
	}
	return false
}

// Returns true if the error was sent, or false if watcher is closed.
func (w *Watcher) sendError(err error) bool {
	select {
	case w.Errors <- err:
		return true
	case <-w.done:
	}
	return false
}

// Close removes all watches and closes the events channel.
func (w *Watcher) Close() error {
	w.mu.Lock()
	if w.isClosed {
		w.mu.Unlock()
		return nil
	}
	w.isClosed = true

	// copy paths to remove while locked
	pathsToRemove := make([]string, 0, len(w.watches))
	for name := range w.watches {
		pathsToRemove = append(pathsToRemove, name)
	}
	w.mu.Unlock() // Unlock before calling Remove, which also locks
	for _, name := range pathsToRemove {
		w.Remove(name)
	}

	// Send "quit" message to the reader goroutine.
	unix.Close(w.closepipe[1])
	close(w.done)

	return nil
}

// Add starts monitoring the path for changes.
//
// A path can only be watched once; attempting to watch it more than once will
// return an error. Paths that do not yet exist on the filesystem cannot be
// added. A watch will be automatically removed if the path is deleted.
//
// A path will remain watched if it gets renamed to somewhere else on the same
// filesystem, but the monitor will get removed if the path gets deleted and
// re-created, or if it's moved to a different filesystem.
//
// Notifications on network filesystems (NFS, SMB, FUSE, etc.) or special
// filesystems (/proc, /sys, etc.) generally don't work.
//
// # Watching directories
//
// All files in a directory are monitored, including new files that are created
// after the watcher is started. Subdirectories are not watched (i.e. it's
// non-recursive).
//
// # Watching files
//
// Watching individual files (rather than directories) is generally not
// recommended as many tools update files atomically. Instead of "just" writing
// to the file a temporary file will be written to first, and if successful the
// temporary file is moved to to destination removing the original, or some
// variant thereof. The watcher on the original file is now lost, as it no
// longer exists.
//
// Instead, watch the parent directory and use Event.Name to filter out files
// you're not interested in. There is an example of this in [cmd/fsnotify/file.go].
func (w *Watcher) Add(name string) error {
	w.mu.Lock()
	w.userWatches[name] = struct{}{}
	w.mu.Unlock()
	_, err := w.addWatch(name, noteAllEvents)
	return err
}

// Remove stops monitoring the path for changes.
//
// Directories are always removed non-recursively. For example, if you added
// /tmp/dir and /tmp/dir/subdir then you will need to remove both.
//
// Removing a path that has not yet been added returns [ErrNonExistentWatch].
func (w *Watcher) Remove(name string) error {
	name = filepath.Clean(name)
	w.mu.Lock()
	watchfd, ok := w.watches[name]
	w.mu.Unlock()
	if !ok {
		return fmt.Errorf("%w: %s", ErrNonExistentWatch, name)
	}

	err := w.register([]int{watchfd}, unix.EV_DELETE, 0)
	if err != nil {
		return err
	}

	unix.Close(watchfd)

	w.mu.Lock()
	isDir := w.paths[watchfd].isDir
	delete(w.watches, name)
	delete(w.userWatches, name)

	parentName := filepath.Dir(name)
	delete(w.watchesByDir[parentName], watchfd)

	if len(w.watchesByDir[parentName]) == 0 {
		delete(w.watchesByDir, parentName)
	}

	delete(w.paths, watchfd)
	delete(w.dirFlags, name)
	delete(w.fileExists, name)
	w.mu.Unlock()

	// Find all watched paths that are in this directory that are not external.
	if isDir {
		var pathsToRemove []string
		w.mu.Lock()
		for fd := range w.watchesByDir[name] {
			path := w.paths[fd]
			if _, ok := w.userWatches[path.name]; !ok {
				pathsToRemove = append(pathsToRemove, path.name)
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

// WatchList returns all paths added with [Add] (and are not yet removed).
func (w *Watcher) WatchList() []string {
	w.mu.Lock()
	defer w.mu.Unlock()

	entries := make([]string, 0, len(w.userWatches))
	for pathname := range w.userWatches {
		entries = append(entries, pathname)
	}

	return entries
}

// Watch all events (except NOTE_EXTEND, NOTE_LINK, NOTE_REVOKE)
const noteAllEvents = unix.NOTE_DELETE | unix.NOTE_WRITE | unix.NOTE_ATTRIB | unix.NOTE_RENAME

// addWatch adds name to the watched file set.
// The flags are interpreted as described in kevent(2).
// Returns the real path to the file which was added, if any, which may be different from the one passed in the case of symlinks.
func (w *Watcher) addWatch(name string, flags uint32) (string, error) {
	var isDir bool
	// Make ./name and name equivalent
	name = filepath.Clean(name)

	w.mu.Lock()
	if w.isClosed {
		w.mu.Unlock()
		return "", errors.New("kevent instance already closed")
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
			return "", err
		}

		// Don't watch sockets or named pipes
		if (fi.Mode()&os.ModeSocket == os.ModeSocket) || (fi.Mode()&os.ModeNamedPipe == os.ModeNamedPipe) {
			return "", nil
		}

		// Follow Symlinks
		//
		// Linux can add unresolvable symlinks to the watch list without issue,
		// and Windows can't do symlinks period. To maintain consistency, we
		// will act like everything is fine if the link can't be resolved.
		// There will simply be no file events for broken symlinks. Hence the
		// returns of nil on errors.
		if fi.Mode()&os.ModeSymlink == os.ModeSymlink {
			name, err = filepath.EvalSymlinks(name)
			if err != nil {
				return "", nil
			}

			w.mu.Lock()
			_, alreadyWatching = w.watches[name]
			w.mu.Unlock()

			if alreadyWatching {
				return name, nil
			}

			fi, err = os.Lstat(name)
			if err != nil {
				return "", nil
			}
		}

		// Retry on EINTR; open() can return EINTR in practice on macOS.
		// See #354, and go issues 11180 and 39237.
		for {
			watchfd, err = unix.Open(name, openMode, 0)
			if err == nil {
				break
			}
			if errors.Is(err, unix.EINTR) {
				continue
			}

			return "", err
		}

		isDir = fi.IsDir()
	}

	err := w.register([]int{watchfd}, unix.EV_ADD|unix.EV_CLEAR|unix.EV_ENABLE, flags)
	if err != nil {
		unix.Close(watchfd)
		return "", err
	}

	if !alreadyWatching {
		w.mu.Lock()
		parentName := filepath.Dir(name)
		w.watches[name] = watchfd

		watchesByDir, ok := w.watchesByDir[parentName]
		if !ok {
			watchesByDir = make(map[int]struct{}, 1)
			w.watchesByDir[parentName] = watchesByDir
		}
		watchesByDir[watchfd] = struct{}{}

		w.paths[watchfd] = pathInfo{name: name, isDir: isDir}
		w.mu.Unlock()
	}

	if isDir {
		// Watch the directory if it has not been watched before,
		// or if it was watched before, but perhaps only a NOTE_DELETE (watchDirectoryFiles)
		w.mu.Lock()

		watchDir := (flags&unix.NOTE_WRITE) == unix.NOTE_WRITE &&
			(!alreadyWatching || (w.dirFlags[name]&unix.NOTE_WRITE) != unix.NOTE_WRITE)
		// Store flags so this watch can be updated later
		w.dirFlags[name] = flags
		w.mu.Unlock()

		if watchDir {
			if err := w.watchDirectoryFiles(name); err != nil {
				return "", err
			}
		}
	}
	return name, nil
}

// readEvents reads from kqueue and converts the received kevents into
// Event values that it sends down the Events channel.
func (w *Watcher) readEvents() {
	defer func() {
		err := unix.Close(w.kq)
		if err != nil {
			w.Errors <- err
		}
		unix.Close(w.closepipe[0])
		close(w.Events)
		close(w.Errors)
	}()

	eventBuffer := make([]unix.Kevent_t, 10)
	for closed := false; !closed; {
		kevents, err := w.read(eventBuffer)
		// EINTR is okay, the syscall was interrupted before timeout expired.
		if err != nil && err != unix.EINTR {
			if !w.sendError(fmt.Errorf("fsnotify.readEvents: %w", err)) {
				closed = true
			}
			continue
		}

		// Flush the events we received to the Events channel
		for _, kevent := range kevents {
			var (
				watchfd = int(kevent.Ident)
				mask    = uint32(kevent.Fflags)
			)

			// Shut down the loop when the pipe is closed, but only after all
			// other events have been processed.
			if watchfd == w.closepipe[0] {
				closed = true
				continue
			}

			w.mu.Lock()
			path := w.paths[watchfd]
			w.mu.Unlock()

			event := w.newEvent(path.name, mask)

			if path.isDir && !event.Has(Remove) {
				// Double check to make sure the directory exists. This can
				// happen when we do a rm -fr on a recursively watched folders
				// and we receive a modification event first but the folder has
				// been deleted and later receive the delete event.
				if _, err := os.Lstat(event.Name); os.IsNotExist(err) {
					event.Op |= Remove
				}
			}

			if event.Has(Rename) || event.Has(Remove) {
				w.Remove(event.Name)
				w.mu.Lock()
				delete(w.fileExists, event.Name)
				w.mu.Unlock()
			}

			if path.isDir && event.Has(Write) && !event.Has(Remove) {
				w.sendDirectoryChangeEvents(event.Name)
			} else {
				if !w.sendEvent(event) {
					closed = true
					continue
				}
			}

			if event.Has(Remove) {
				// Look for a file that may have overwritten this.
				// For example, mv f1 f2 will delete f2, then create f2.
				if path.isDir {
					fileDir := filepath.Clean(event.Name)
					w.mu.Lock()
					_, found := w.watches[fileDir]
					w.mu.Unlock()
					if found {
						// make sure the directory exists before we watch for changes. When we
						// do a recursive watch and perform rm -fr, the parent directory might
						// have gone missing, ignore the missing directory and let the
						// upcoming delete event remove the watch from the parent directory.
						if _, err := os.Lstat(fileDir); err == nil {
							w.sendDirectoryChangeEvents(fileDir)
						}
					}
				} else {
					filePath := filepath.Clean(event.Name)
					if fileInfo, err := os.Lstat(filePath); err == nil {
						w.sendFileCreatedEventIfNew(filePath, fileInfo)
					}
				}
			}
		}
	}
}

// newEvent returns an platform-independent Event based on kqueue Fflags.
func (w *Watcher) newEvent(name string, mask uint32) Event {
	e := Event{Name: name}
	if mask&unix.NOTE_DELETE == unix.NOTE_DELETE {
		e.Op |= Remove
	}
	if mask&unix.NOTE_WRITE == unix.NOTE_WRITE {
		e.Op |= Write
	}
	if mask&unix.NOTE_RENAME == unix.NOTE_RENAME {
		e.Op |= Rename
	}
	if mask&unix.NOTE_ATTRIB == unix.NOTE_ATTRIB {
		e.Op |= Chmod
	}
	return e
}

// watchDirectoryFiles to mimic inotify when adding a watch on a directory
func (w *Watcher) watchDirectoryFiles(dirPath string) error {
	// Get all files
	files, err := ioutil.ReadDir(dirPath)
	if err != nil {
		return err
	}

	for _, fileInfo := range files {
		path := filepath.Join(dirPath, fileInfo.Name())

		cleanPath, err := w.internalWatch(path, fileInfo)
		if err != nil {
			// No permission to read the file; that's not a problem: just skip.
			// But do add it to w.fileExists to prevent it from being picked up
			// as a "new" file later (it still shows up in the directory
			// listing).
			switch {
			case errors.Is(err, unix.EACCES) || errors.Is(err, unix.EPERM):
				cleanPath = filepath.Clean(path)
			default:
				return fmt.Errorf("%q: %w", filepath.Join(dirPath, fileInfo.Name()), err)
			}
		}

		w.mu.Lock()
		w.fileExists[cleanPath] = struct{}{}
		w.mu.Unlock()
	}

	return nil
}

// Search the directory for new files and send an event for them.
//
// This functionality is to have the BSD watcher match the inotify, which sends
// a create event for files created in a watched directory.
func (w *Watcher) sendDirectoryChangeEvents(dir string) {
	// Get all files
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		if !w.sendError(fmt.Errorf("fsnotify.sendDirectoryChangeEvents: %w", err)) {
			return
		}
	}

	// Search for new files
	for _, fi := range files {
		err := w.sendFileCreatedEventIfNew(filepath.Join(dir, fi.Name()), fi)
		if err != nil {
			return
		}
	}
}

// sendFileCreatedEvent sends a create event if the file isn't already being tracked.
func (w *Watcher) sendFileCreatedEventIfNew(filePath string, fileInfo os.FileInfo) (err error) {
	w.mu.Lock()
	_, doesExist := w.fileExists[filePath]
	w.mu.Unlock()
	if !doesExist {
		if !w.sendEvent(Event{Name: filePath, Op: Create}) {
			return
		}
	}

	// like watchDirectoryFiles (but without doing another ReadDir)
	filePath, err = w.internalWatch(filePath, fileInfo)
	if err != nil {
		return err
	}

	w.mu.Lock()
	w.fileExists[filePath] = struct{}{}
	w.mu.Unlock()

	return nil
}

func (w *Watcher) internalWatch(name string, fileInfo os.FileInfo) (string, error) {
	if fileInfo.IsDir() {
		// mimic Linux providing delete events for subdirectories
		// but preserve the flags used if currently watching subdirectory
		w.mu.Lock()
		flags := w.dirFlags[name]
		w.mu.Unlock()

		flags |= unix.NOTE_DELETE | unix.NOTE_RENAME
		return w.addWatch(name, flags)
	}

	// watch file to mimic Linux inotify
	return w.addWatch(name, noteAllEvents)
}

// Register events with the queue.
func (w *Watcher) register(fds []int, flags int, fflags uint32) error {
	changes := make([]unix.Kevent_t, len(fds))
	for i, fd := range fds {
		// SetKevent converts int to the platform-specific types.
		unix.SetKevent(&changes[i], fd, unix.EVFILT_VNODE, flags)
		changes[i].Fflags = fflags
	}

	// Register the events.
	success, err := unix.Kevent(w.kq, changes, nil, nil)
	if success == -1 {
		return err
	}
	return nil
}

// read retrieves pending events, or waits until an event occurs.
func (w *Watcher) read(events []unix.Kevent_t) ([]unix.Kevent_t, error) {
	n, err := unix.Kevent(w.kq, nil, events, nil)
	if err != nil {
		return nil, err
	}
	return events[0:n], nil
}
