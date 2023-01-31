#!/usr/bin/env zsh
[ "${ZSH_VERSION:-}" = "" ] && echo >&2 "Only works with zsh" && exit 1
setopt err_exit no_unset pipefail extended_glob

# Simple script to update the godoc comments on all watchers. Probably took me
# more time to write this than doing it manually, but ah well 🙃

watcher=$(<<EOF
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
EOF
)

new=$(<<EOF
// NewWatcher creates a new Watcher.
EOF
)

add=$(<<EOF
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
EOF
)

remove=$(<<EOF
// Remove stops monitoring the path for changes.
//
// Directories are always removed non-recursively. For example, if you added
// /tmp/dir and /tmp/dir/subdir then you will need to remove both.
//
// Removing a path that has not yet been added returns [ErrNonExistentWatch].
EOF
)

close=$(<<EOF
// Close removes all watches and closes the events channel.
EOF
)

watchlist=$(<<EOF
// WatchList returns all paths added with [Add] (and are not yet removed).
EOF
)

events=$(<<EOF
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
EOF
)

errors=$(<<EOF
	// Errors sends any errors.
EOF
)

set-cmt() {
	local pat=$1
	local cmt=$2

	IFS=$'\n' local files=($(grep -n $pat backend_*~*_test.go))
	for f in $files; do
		IFS=':' local fields=($=f)
		local file=$fields[1]
		local end=$(( $fields[2] - 1 ))

		# Find start of comment.
		local start=0
		IFS=$'\n' local lines=($(head -n$end $file))
		for (( i = 1; i <= $#lines; i++ )); do
			local line=$lines[-$i]
			if ! grep -q '^[[:space:]]*//' <<<$line; then
				start=$(( end - (i - 2) ))
				break
			fi
		done

		head -n $(( start - 1 )) $file  >/tmp/x
		print -r -- $cmt                >>/tmp/x
		tail -n+$(( end + 1 ))   $file  >>/tmp/x
		mv /tmp/x $file
	done
}

set-cmt '^type Watcher struct '             $watcher
set-cmt '^func NewWatcher('                 $new
set-cmt '^func (w \*Watcher) Add('          $add
set-cmt '^func (w \*Watcher) Remove('       $remove
set-cmt '^func (w \*Watcher) Close('        $close
set-cmt '^func (w \*Watcher) WatchList('    $watchlist
set-cmt '^[[:space:]]*Events *chan Event$'  $events
set-cmt '^[[:space:]]*Errors *chan error$'  $errors
