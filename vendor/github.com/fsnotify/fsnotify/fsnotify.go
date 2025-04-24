// Package fsnotify provides a cross-platform interface for file system
// notifications.
//
// Currently supported systems:
//
//   - Linux      via inotify
//   - BSD, macOS via kqueue
//   - Windows    via ReadDirectoryChangesW
//   - illumos    via FEN
//
// # FSNOTIFY_DEBUG
//
// Set the FSNOTIFY_DEBUG environment variable to "1" to print debug messages to
// stderr. This can be useful to track down some problems, especially in cases
// where fsnotify is used as an indirect dependency.
//
// Every event will be printed as soon as there's something useful to print,
// with as little processing from fsnotify.
//
// Example output:
//
//	FSNOTIFY_DEBUG: 11:34:23.633087586   256:IN_CREATE            → "/tmp/file-1"
//	FSNOTIFY_DEBUG: 11:34:23.633202319     4:IN_ATTRIB            → "/tmp/file-1"
//	FSNOTIFY_DEBUG: 11:34:28.989728764   512:IN_DELETE            → "/tmp/file-1"
package fsnotify

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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
// Paths can be added as "C:\\path\\to\\dir", but forward slashes
// ("C:/path/to/dir") will also work.
//
// When a watched directory is removed it will always send an event for the
// directory itself, but may not send events for all files in that directory.
// Sometimes it will send events for all files, sometimes it will send no
// events, and often only for some files.
//
// The default ReadDirectoryChangesW() buffer size is 64K, which is the largest
// value that is guaranteed to work with SMB filesystems. If you have many
// events in quick succession this may not be enough, and you will have to use
// [WithBufferSize] to increase the value.
type Watcher struct {
	b backend

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
	Errors chan error
}

// Event represents a file system notification.
type Event struct {
	// Path to the file or directory.
	//
	// Paths are relative to the input; for example with Add("dir") the Name
	// will be set to "dir/file" if you create that file, but if you use
	// Add("/path/to/dir") it will be "/path/to/dir/file".
	Name string

	// File operation that triggered the event.
	//
	// This is a bitmask and some systems may send multiple operations at once.
	// Use the Event.Has() method instead of comparing with ==.
	Op Op

	// Create events will have this set to the old path if it's a rename. This
	// only works when both the source and destination are watched. It's not
	// reliable when watching individual files, only directories.
	//
	// For example "mv /tmp/file /tmp/rename" will emit:
	//
	//   Event{Op: Rename, Name: "/tmp/file"}
	//   Event{Op: Create, Name: "/tmp/rename", RenamedFrom: "/tmp/file"}
	renamedFrom string
}

// Op describes a set of file operations.
type Op uint32

// The operations fsnotify can trigger; see the documentation on [Watcher] for a
// full description, and check them with [Event.Has].
const (
	// A new pathname was created.
	Create Op = 1 << iota

	// The pathname was written to; this does *not* mean the write has finished,
	// and a write can be followed by more writes.
	Write

	// The path was removed; any watches on it will be removed. Some "remove"
	// operations may trigger a Rename if the file is actually moved (for
	// example "remove to trash" is often a rename).
	Remove

	// The path was renamed to something else; any watches on it will be
	// removed.
	Rename

	// File attributes were changed.
	//
	// It's generally not recommended to take action on this event, as it may
	// get triggered very frequently by some software. For example, Spotlight
	// indexing on macOS, anti-virus software, backup software, etc.
	Chmod

	// File descriptor was opened.
	//
	// Only works on Linux and FreeBSD.
	xUnportableOpen

	// File was read from.
	//
	// Only works on Linux and FreeBSD.
	xUnportableRead

	// File opened for writing was closed.
	//
	// Only works on Linux and FreeBSD.
	//
	// The advantage of using this over Write is that it's more reliable than
	// waiting for Write events to stop. It's also faster (if you're not
	// listening to Write events): copying a file of a few GB can easily
	// generate tens of thousands of Write events in a short span of time.
	xUnportableCloseWrite

	// File opened for reading was closed.
	//
	// Only works on Linux and FreeBSD.
	xUnportableCloseRead
)

var (
	// ErrNonExistentWatch is used when Remove() is called on a path that's not
	// added.
	ErrNonExistentWatch = errors.New("fsnotify: can't remove non-existent watch")

	// ErrClosed is used when trying to operate on a closed Watcher.
	ErrClosed = errors.New("fsnotify: watcher already closed")

	// ErrEventOverflow is reported from the Errors channel when there are too
	// many events:
	//
	//  - inotify:      inotify returns IN_Q_OVERFLOW – because there are too
	//                  many queued events (the fs.inotify.max_queued_events
	//                  sysctl can be used to increase this).
	//  - windows:      The buffer size is too small; WithBufferSize() can be used to increase it.
	//  - kqueue, fen:  Not used.
	ErrEventOverflow = errors.New("fsnotify: queue or buffer overflow")

	// ErrUnsupported is returned by AddWith() when WithOps() specified an
	// Unportable event that's not supported on this platform.
	//lint:ignore ST1012 not relevant
	xErrUnsupported = errors.New("fsnotify: not supported with this backend")
)

// NewWatcher creates a new Watcher.
func NewWatcher() (*Watcher, error) {
	ev, errs := make(chan Event, defaultBufferSize), make(chan error)
	b, err := newBackend(ev, errs)
	if err != nil {
		return nil, err
	}
	return &Watcher{b: b, Events: ev, Errors: errs}, nil
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
	ev, errs := make(chan Event, sz), make(chan error)
	b, err := newBackend(ev, errs)
	if err != nil {
		return nil, err
	}
	return &Watcher{b: b, Events: ev, Errors: errs}, nil
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
// will write to a temporary file which is then moved to destination,
// overwriting the original (or some variant thereof). The watcher on the
// original file is now lost, as that no longer exists.
//
// The upshot of this is that a power failure or crash won't leave a
// half-written file.
//
// Watch the parent directory and use Event.Name to filter out files you're not
// interested in. There is an example of this in cmd/fsnotify/file.go.
func (w *Watcher) Add(path string) error { return w.b.Add(path) }

// AddWith is like [Watcher.Add], but allows adding options. When using Add()
// the defaults described below are used.
//
// Possible options are:
//
//   - [WithBufferSize] sets the buffer size for the Windows backend; no-op on
//     other platforms. The default is 64K (65536 bytes).
func (w *Watcher) AddWith(path string, opts ...addOpt) error { return w.b.AddWith(path, opts...) }

// Remove stops monitoring the path for changes.
//
// Directories are always removed non-recursively. For example, if you added
// /tmp/dir and /tmp/dir/subdir then you will need to remove both.
//
// Removing a path that has not yet been added returns [ErrNonExistentWatch].
//
// Returns nil if [Watcher.Close] was called.
func (w *Watcher) Remove(path string) error { return w.b.Remove(path) }

// Close removes all watches and closes the Events channel.
func (w *Watcher) Close() error { return w.b.Close() }

// WatchList returns all paths explicitly added with [Watcher.Add] (and are not
// yet removed).
//
// The order is undefined, and may differ per call. Returns nil if
// [Watcher.Close] was called.
func (w *Watcher) WatchList() []string { return w.b.WatchList() }

// Supports reports if all the listed operations are supported by this platform.
//
// Create, Write, Remove, Rename, and Chmod are always supported. It can only
// return false for an Op starting with Unportable.
func (w *Watcher) xSupports(op Op) bool { return w.b.xSupports(op) }

func (o Op) String() string {
	var b strings.Builder
	if o.Has(Create) {
		b.WriteString("|CREATE")
	}
	if o.Has(Remove) {
		b.WriteString("|REMOVE")
	}
	if o.Has(Write) {
		b.WriteString("|WRITE")
	}
	if o.Has(xUnportableOpen) {
		b.WriteString("|OPEN")
	}
	if o.Has(xUnportableRead) {
		b.WriteString("|READ")
	}
	if o.Has(xUnportableCloseWrite) {
		b.WriteString("|CLOSE_WRITE")
	}
	if o.Has(xUnportableCloseRead) {
		b.WriteString("|CLOSE_READ")
	}
	if o.Has(Rename) {
		b.WriteString("|RENAME")
	}
	if o.Has(Chmod) {
		b.WriteString("|CHMOD")
	}
	if b.Len() == 0 {
		return "[no events]"
	}
	return b.String()[1:]
}

// Has reports if this operation has the given operation.
func (o Op) Has(h Op) bool { return o&h != 0 }

// Has reports if this event has the given operation.
func (e Event) Has(op Op) bool { return e.Op.Has(op) }

// String returns a string representation of the event with their path.
func (e Event) String() string {
	if e.renamedFrom != "" {
		return fmt.Sprintf("%-13s %q ← %q", e.Op.String(), e.Name, e.renamedFrom)
	}
	return fmt.Sprintf("%-13s %q", e.Op.String(), e.Name)
}

type (
	backend interface {
		Add(string) error
		AddWith(string, ...addOpt) error
		Remove(string) error
		WatchList() []string
		Close() error
		xSupports(Op) bool
	}
	addOpt   func(opt *withOpts)
	withOpts struct {
		bufsize    int
		op         Op
		noFollow   bool
		sendCreate bool
	}
)

var debug = func() bool {
	// Check for exactly "1" (rather than mere existence) so we can add
	// options/flags in the future. I don't know if we ever want that, but it's
	// nice to leave the option open.
	return os.Getenv("FSNOTIFY_DEBUG") == "1"
}()

var defaultOpts = withOpts{
	bufsize: 65536, // 64K
	op:      Create | Write | Remove | Rename | Chmod,
}

func getOptions(opts ...addOpt) withOpts {
	with := defaultOpts
	for _, o := range opts {
		if o != nil {
			o(&with)
		}
	}
	return with
}

// WithBufferSize sets the [ReadDirectoryChangesW] buffer size.
//
// This only has effect on Windows systems, and is a no-op for other backends.
//
// The default value is 64K (65536 bytes) which is the highest value that works
// on all filesystems and should be enough for most applications, but if you
// have a large burst of events it may not be enough. You can increase it if
// you're hitting "queue or buffer overflow" errors ([ErrEventOverflow]).
//
// [ReadDirectoryChangesW]: https://learn.microsoft.com/en-gb/windows/win32/api/winbase/nf-winbase-readdirectorychangesw
func WithBufferSize(bytes int) addOpt {
	return func(opt *withOpts) { opt.bufsize = bytes }
}

// WithOps sets which operations to listen for. The default is [Create],
// [Write], [Remove], [Rename], and [Chmod].
//
// Excluding operations you're not interested in can save quite a bit of CPU
// time; in some use cases there may be hundreds of thousands of useless Write
// or Chmod operations per second.
//
// This can also be used to add unportable operations not supported by all
// platforms; unportable operations all start with "Unportable":
// [UnportableOpen], [UnportableRead], [UnportableCloseWrite], and
// [UnportableCloseRead].
//
// AddWith returns an error when using an unportable operation that's not
// supported. Use [Watcher.Support] to check for support.
func withOps(op Op) addOpt {
	return func(opt *withOpts) { opt.op = op }
}

// WithNoFollow disables following symlinks, so the symlinks themselves are
// watched.
func withNoFollow() addOpt {
	return func(opt *withOpts) { opt.noFollow = true }
}

// "Internal" option for recursive watches on inotify.
func withCreate() addOpt {
	return func(opt *withOpts) { opt.sendCreate = true }
}

var enableRecurse = false

// Check if this path is recursive (ends with "/..." or "\..."), and return the
// path with the /... stripped.
func recursivePath(path string) (string, bool) {
	path = filepath.Clean(path)
	if !enableRecurse { // Only enabled in tests for now.
		return path, false
	}
	if filepath.Base(path) == "..." {
		return filepath.Dir(path), true
	}
	return path, false
}
