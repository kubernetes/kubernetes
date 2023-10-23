//go:build solaris
// +build solaris

// Note: the documentation on the Watcher type and methods is generated from
// mkdoc.zsh

package fsnotify

import (
	"errors"
	"fmt"
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
//	fp := os.Open("file")
//	os.Remove("file")        // Triggers Chmod
//	fp.Close()               // Triggers Remove
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
//	# Default values on Linux 5.18
//	sysctl fs.inotify.max_user_watches=124983
//	sysctl fs.inotify.max_user_instances=128
//
// To make the changes persist on reboot edit /etc/sysctl.conf or
// /usr/lib/sysctl.d/50-default.conf (details differ per Linux distro; check
// your distro's documentation):
//
//	fs.inotify.max_user_watches=124983
//	fs.inotify.max_user_instances=128
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
// # Windows notes
//
// Paths can be added as "C:\path\to\dir", but forward slashes
// ("C:/path/to/dir") will also work.
//
// When a watched directory is removed it will always send an event for the
// directory itself, but may not send events for all files in that directory.
// Sometimes it will send events for all times, sometimes it will send no
// events, and often only for some files.
//
// The default ReadDirectoryChangesW() buffer size is 64K, which is the largest
// value that is guaranteed to work with SMB filesystems. If you have many
// events in quick succession this may not be enough, and you will have to use
// [WithBufferSize] to increase the value.
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
	//                      you may get hundreds of Write events, and you may
	//                      want to wait until you've stopped receiving them
	//                      (see the dedup example in cmd/fsnotify).
	//
	//                      Some systems may send Write event for directories
	//                      when the directory content changes.
	//
	//   fsnotify.Chmod     Attributes were changed. On Linux this is also sent
	//                      when a file is removed (or more accurately, when a
	//                      link to an inode is removed). On kqueue it's sent
	//                      when a file is truncated. On Windows it's never
	//                      sent.
	Events chan Event

	// Errors sends any errors.
	//
	// ErrEventOverflow is used to indicate there are too many events:
	//
	//  - inotify:      There are too many queued events (fs.inotify.max_queued_events sysctl)
	//  - windows:      The buffer size is too small; WithBufferSize() can be used to increase it.
	//  - kqueue, fen:  Not used.
	Errors chan error

	mu      sync.Mutex
	port    *unix.EventPort
	done    chan struct{}       // Channel for sending a "quit message" to the reader goroutine
	dirs    map[string]struct{} // Explicitly watched directories
	watches map[string]struct{} // Explicitly watched non-directories
}

// NewWatcher creates a new Watcher.
func NewWatcher() (*Watcher, error) {
	return NewBufferedWatcher(0)
}

// NewBufferedWatcher creates a new Watcher with a buffered Watcher.Events
// channel.
//
// The main use case for this is situations with a very large number of events
// where the kernel buffer size can't be increased (e.g. due to lack of
// permissions). An unbuffered Watcher will perform better for almost all use
// cases, and whenever possible you will be better off increasing the kernel
// buffers instead of adding a large userspace buffer.
func NewBufferedWatcher(sz uint) (*Watcher, error) {
	w := &Watcher{
		Events:  make(chan Event, sz),
		Errors:  make(chan error),
		dirs:    make(map[string]struct{}),
		watches: make(map[string]struct{}),
		done:    make(chan struct{}),
	}

	var err error
	w.port, err = unix.NewEventPort()
	if err != nil {
		return nil, fmt.Errorf("fsnotify.NewWatcher: %w", err)
	}

	go w.readEvents()
	return w, nil
}

// sendEvent attempts to send an event to the user, returning true if the event
// was put in the channel successfully and false if the watcher has been closed.
func (w *Watcher) sendEvent(name string, op Op) (sent bool) {
	select {
	case w.Events <- Event{Name: name, Op: op}:
		return true
	case <-w.done:
		return false
	}
}

// sendError attempts to send an error to the user, returning true if the error
// was put in the channel successfully and false if the watcher has been closed.
func (w *Watcher) sendError(err error) (sent bool) {
	select {
	case w.Errors <- err:
		return true
	case <-w.done:
		return false
	}
}

func (w *Watcher) isClosed() bool {
	select {
	case <-w.done:
		return true
	default:
		return false
	}
}

// Close removes all watches and closes the Events channel.
func (w *Watcher) Close() error {
	// Take the lock used by associateFile to prevent lingering events from
	// being processed after the close
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.isClosed() {
		return nil
	}
	close(w.done)
	return w.port.Close()
}

// Add starts monitoring the path for changes.
//
// A path can only be watched once; watching it more than once is a no-op and will
// not return an error. Paths that do not yet exist on the filesystem cannot be
// watched.
//
// A watch will be automatically removed if the watched path is deleted or
// renamed. The exception is the Windows backend, which doesn't remove the
// watcher on renames.
//
// Notifications on network filesystems (NFS, SMB, FUSE, etc.) or special
// filesystems (/proc, /sys, etc.) generally don't work.
//
// Returns [ErrClosed] if [Watcher.Close] was called.
//
// See [Watcher.AddWith] for a version that allows adding options.
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
// recommended as many programs (especially editors) update files atomically: it
// will write to a temporary file which is then moved to to destination,
// overwriting the original (or some variant thereof). The watcher on the
// original file is now lost, as that no longer exists.
//
// The upshot of this is that a power failure or crash won't leave a
// half-written file.
//
// Watch the parent directory and use Event.Name to filter out files you're not
// interested in. There is an example of this in cmd/fsnotify/file.go.
func (w *Watcher) Add(name string) error { return w.AddWith(name) }

// AddWith is like [Watcher.Add], but allows adding options. When using Add()
// the defaults described below are used.
//
// Possible options are:
//
//   - [WithBufferSize] sets the buffer size for the Windows backend; no-op on
//     other platforms. The default is 64K (65536 bytes).
func (w *Watcher) AddWith(name string, opts ...addOpt) error {
	if w.isClosed() {
		return ErrClosed
	}
	if w.port.PathIsWatched(name) {
		return nil
	}

	_ = getOptions(opts...)

	// Currently we resolve symlinks that were explicitly requested to be
	// watched. Otherwise we would use LStat here.
	stat, err := os.Stat(name)
	if err != nil {
		return err
	}

	// Associate all files in the directory.
	if stat.IsDir() {
		err := w.handleDirectory(name, stat, true, w.associateFile)
		if err != nil {
			return err
		}

		w.mu.Lock()
		w.dirs[name] = struct{}{}
		w.mu.Unlock()
		return nil
	}

	err = w.associateFile(name, stat, true)
	if err != nil {
		return err
	}

	w.mu.Lock()
	w.watches[name] = struct{}{}
	w.mu.Unlock()
	return nil
}

// Remove stops monitoring the path for changes.
//
// Directories are always removed non-recursively. For example, if you added
// /tmp/dir and /tmp/dir/subdir then you will need to remove both.
//
// Removing a path that has not yet been added returns [ErrNonExistentWatch].
//
// Returns nil if [Watcher.Close] was called.
func (w *Watcher) Remove(name string) error {
	if w.isClosed() {
		return nil
	}
	if !w.port.PathIsWatched(name) {
		return fmt.Errorf("%w: %s", ErrNonExistentWatch, name)
	}

	// The user has expressed an intent. Immediately remove this name from
	// whichever watch list it might be in. If it's not in there the delete
	// doesn't cause harm.
	w.mu.Lock()
	delete(w.watches, name)
	delete(w.dirs, name)
	w.mu.Unlock()

	stat, err := os.Stat(name)
	if err != nil {
		return err
	}

	// Remove associations for every file in the directory.
	if stat.IsDir() {
		err := w.handleDirectory(name, stat, false, w.dissociateFile)
		if err != nil {
			return err
		}
		return nil
	}

	err = w.port.DissociatePath(name)
	if err != nil {
		return err
	}

	return nil
}

// readEvents contains the main loop that runs in a goroutine watching for events.
func (w *Watcher) readEvents() {
	// If this function returns, the watcher has been closed and we can close
	// these channels
	defer func() {
		close(w.Errors)
		close(w.Events)
	}()

	pevents := make([]unix.PortEvent, 8)
	for {
		count, err := w.port.Get(pevents, 1, nil)
		if err != nil && err != unix.ETIME {
			// Interrupted system call (count should be 0) ignore and continue
			if errors.Is(err, unix.EINTR) && count == 0 {
				continue
			}
			// Get failed because we called w.Close()
			if errors.Is(err, unix.EBADF) && w.isClosed() {
				return
			}
			// There was an error not caused by calling w.Close()
			if !w.sendError(err) {
				return
			}
		}

		p := pevents[:count]
		for _, pevent := range p {
			if pevent.Source != unix.PORT_SOURCE_FILE {
				// Event from unexpected source received; should never happen.
				if !w.sendError(errors.New("Event from unexpected source received")) {
					return
				}
				continue
			}

			err = w.handleEvent(&pevent)
			if err != nil {
				if !w.sendError(err) {
					return
				}
			}
		}
	}
}

func (w *Watcher) handleDirectory(path string, stat os.FileInfo, follow bool, handler func(string, os.FileInfo, bool) error) error {
	files, err := os.ReadDir(path)
	if err != nil {
		return err
	}

	// Handle all children of the directory.
	for _, entry := range files {
		finfo, err := entry.Info()
		if err != nil {
			return err
		}
		err = handler(filepath.Join(path, finfo.Name()), finfo, false)
		if err != nil {
			return err
		}
	}

	// And finally handle the directory itself.
	return handler(path, stat, follow)
}

// handleEvent might need to emit more than one fsnotify event if the events
// bitmap matches more than one event type (e.g. the file was both modified and
// had the attributes changed between when the association was created and the
// when event was returned)
func (w *Watcher) handleEvent(event *unix.PortEvent) error {
	var (
		events     = event.Events
		path       = event.Path
		fmode      = event.Cookie.(os.FileMode)
		reRegister = true
	)

	w.mu.Lock()
	_, watchedDir := w.dirs[path]
	_, watchedPath := w.watches[path]
	w.mu.Unlock()
	isWatched := watchedDir || watchedPath

	if events&unix.FILE_DELETE != 0 {
		if !w.sendEvent(path, Remove) {
			return nil
		}
		reRegister = false
	}
	if events&unix.FILE_RENAME_FROM != 0 {
		if !w.sendEvent(path, Rename) {
			return nil
		}
		// Don't keep watching the new file name
		reRegister = false
	}
	if events&unix.FILE_RENAME_TO != 0 {
		// We don't report a Rename event for this case, because Rename events
		// are interpreted as referring to the _old_ name of the file, and in
		// this case the event would refer to the new name of the file. This
		// type of rename event is not supported by fsnotify.

		// inotify reports a Remove event in this case, so we simulate this
		// here.
		if !w.sendEvent(path, Remove) {
			return nil
		}
		// Don't keep watching the file that was removed
		reRegister = false
	}

	// The file is gone, nothing left to do.
	if !reRegister {
		if watchedDir {
			w.mu.Lock()
			delete(w.dirs, path)
			w.mu.Unlock()
		}
		if watchedPath {
			w.mu.Lock()
			delete(w.watches, path)
			w.mu.Unlock()
		}
		return nil
	}

	// If we didn't get a deletion the file still exists and we're going to have
	// to watch it again. Let's Stat it now so that we can compare permissions
	// and have what we need to continue watching the file

	stat, err := os.Lstat(path)
	if err != nil {
		// This is unexpected, but we should still emit an event. This happens
		// most often on "rm -r" of a subdirectory inside a watched directory We
		// get a modify event of something happening inside, but by the time we
		// get here, the sudirectory is already gone. Clearly we were watching
		// this path but now it is gone. Let's tell the user that it was
		// removed.
		if !w.sendEvent(path, Remove) {
			return nil
		}
		// Suppress extra write events on removed directories; they are not
		// informative and can be confusing.
		return nil
	}

	// resolve symlinks that were explicitly watched as we would have at Add()
	// time. this helps suppress spurious Chmod events on watched symlinks
	if isWatched {
		stat, err = os.Stat(path)
		if err != nil {
			// The symlink still exists, but the target is gone. Report the
			// Remove similar to above.
			if !w.sendEvent(path, Remove) {
				return nil
			}
			// Don't return the error
		}
	}

	if events&unix.FILE_MODIFIED != 0 {
		if fmode.IsDir() {
			if watchedDir {
				if err := w.updateDirectory(path); err != nil {
					return err
				}
			} else {
				if !w.sendEvent(path, Write) {
					return nil
				}
			}
		} else {
			if !w.sendEvent(path, Write) {
				return nil
			}
		}
	}
	if events&unix.FILE_ATTRIB != 0 && stat != nil {
		// Only send Chmod if perms changed
		if stat.Mode().Perm() != fmode.Perm() {
			if !w.sendEvent(path, Chmod) {
				return nil
			}
		}
	}

	if stat != nil {
		// If we get here, it means we've hit an event above that requires us to
		// continue watching the file or directory
		return w.associateFile(path, stat, isWatched)
	}
	return nil
}

func (w *Watcher) updateDirectory(path string) error {
	// The directory was modified, so we must find unwatched entities and watch
	// them. If something was removed from the directory, nothing will happen,
	// as everything else should still be watched.
	files, err := os.ReadDir(path)
	if err != nil {
		return err
	}

	for _, entry := range files {
		path := filepath.Join(path, entry.Name())
		if w.port.PathIsWatched(path) {
			continue
		}

		finfo, err := entry.Info()
		if err != nil {
			return err
		}
		err = w.associateFile(path, finfo, false)
		if err != nil {
			if !w.sendError(err) {
				return nil
			}
		}
		if !w.sendEvent(path, Create) {
			return nil
		}
	}
	return nil
}

func (w *Watcher) associateFile(path string, stat os.FileInfo, follow bool) error {
	if w.isClosed() {
		return ErrClosed
	}
	// This is primarily protecting the call to AssociatePath but it is
	// important and intentional that the call to PathIsWatched is also
	// protected by this mutex. Without this mutex, AssociatePath has been seen
	// to error out that the path is already associated.
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.port.PathIsWatched(path) {
		// Remove the old association in favor of this one If we get ENOENT,
		// then while the x/sys/unix wrapper still thought that this path was
		// associated, the underlying event port did not. This call will have
		// cleared up that discrepancy. The most likely cause is that the event
		// has fired but we haven't processed it yet.
		err := w.port.DissociatePath(path)
		if err != nil && err != unix.ENOENT {
			return err
		}
	}
	// FILE_NOFOLLOW means we watch symlinks themselves rather than their
	// targets.
	events := unix.FILE_MODIFIED | unix.FILE_ATTRIB | unix.FILE_NOFOLLOW
	if follow {
		// We *DO* follow symlinks for explicitly watched entries.
		events = unix.FILE_MODIFIED | unix.FILE_ATTRIB
	}
	return w.port.AssociatePath(path, stat,
		events,
		stat.Mode())
}

func (w *Watcher) dissociateFile(path string, stat os.FileInfo, unused bool) error {
	if !w.port.PathIsWatched(path) {
		return nil
	}
	return w.port.DissociatePath(path)
}

// WatchList returns all paths explicitly added with [Watcher.Add] (and are not
// yet removed).
//
// Returns nil if [Watcher.Close] was called.
func (w *Watcher) WatchList() []string {
	if w.isClosed() {
		return nil
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	entries := make([]string, 0, len(w.watches)+len(w.dirs))
	for pathname := range w.dirs {
		entries = append(entries, pathname)
	}
	for pathname := range w.watches {
		entries = append(entries, pathname)
	}

	return entries
}
