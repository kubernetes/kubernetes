//go:build appengine || (!darwin && !dragonfly && !freebsd && !openbsd && !linux && !netbsd && !solaris && !windows)
// +build appengine !darwin,!dragonfly,!freebsd,!openbsd,!linux,!netbsd,!solaris,!windows

// Note: the documentation on the Watcher type and methods is generated from
// mkdoc.zsh

package fsnotify

import "errors"

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
}

// NewWatcher creates a new Watcher.
func NewWatcher() (*Watcher, error) {
	return nil, errors.New("fsnotify not supported on the current platform")
}

// NewBufferedWatcher creates a new Watcher with a buffered Watcher.Events
// channel.
//
// The main use case for this is situations with a very large number of events
// where the kernel buffer size can't be increased (e.g. due to lack of
// permissions). An unbuffered Watcher will perform better for almost all use
// cases, and whenever possible you will be better off increasing the kernel
// buffers instead of adding a large userspace buffer.
func NewBufferedWatcher(sz uint) (*Watcher, error) { return NewWatcher() }

// Close removes all watches and closes the Events channel.
func (w *Watcher) Close() error { return nil }

// WatchList returns all paths explicitly added with [Watcher.Add] (and are not
// yet removed).
//
// Returns nil if [Watcher.Close] was called.
func (w *Watcher) WatchList() []string { return nil }

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
func (w *Watcher) Add(name string) error { return nil }

// AddWith is like [Watcher.Add], but allows adding options. When using Add()
// the defaults described below are used.
//
// Possible options are:
//
//   - [WithBufferSize] sets the buffer size for the Windows backend; no-op on
//     other platforms. The default is 64K (65536 bytes).
func (w *Watcher) AddWith(name string, opts ...addOpt) error { return nil }

// Remove stops monitoring the path for changes.
//
// Directories are always removed non-recursively. For example, if you added
// /tmp/dir and /tmp/dir/subdir then you will need to remove both.
//
// Removing a path that has not yet been added returns [ErrNonExistentWatch].
//
// Returns nil if [Watcher.Close] was called.
func (w *Watcher) Remove(name string) error { return nil }
