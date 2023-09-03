//go:build !darwin && !dragonfly && !freebsd && !openbsd && !linux && !netbsd && !solaris && !windows
// +build !darwin,!dragonfly,!freebsd,!openbsd,!linux,!netbsd,!solaris,!windows

package fsnotify

import (
	"fmt"
	"runtime"
)

// Watcher watches a set of files, delivering events to a channel.
type Watcher struct{}

// NewWatcher creates a new Watcher.
func NewWatcher() (*Watcher, error) {
	return nil, fmt.Errorf("fsnotify not supported on %s", runtime.GOOS)
}

// Close removes all watches and closes the events channel.
func (w *Watcher) Close() error {
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
	return nil
}

// Remove stops monitoring the path for changes.
//
// Directories are always removed non-recursively. For example, if you added
// /tmp/dir and /tmp/dir/subdir then you will need to remove both.
//
// Removing a path that has not yet been added returns [ErrNonExistentWatch].
func (w *Watcher) Remove(name string) error {
	return nil
}
