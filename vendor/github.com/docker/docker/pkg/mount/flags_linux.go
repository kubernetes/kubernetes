package mount // import "github.com/docker/docker/pkg/mount"

import (
	"golang.org/x/sys/unix"
)

const (
	// RDONLY will mount the file system read-only.
	RDONLY = unix.MS_RDONLY

	// NOSUID will not allow set-user-identifier or set-group-identifier bits to
	// take effect.
	NOSUID = unix.MS_NOSUID

	// NODEV will not interpret character or block special devices on the file
	// system.
	NODEV = unix.MS_NODEV

	// NOEXEC will not allow execution of any binaries on the mounted file system.
	NOEXEC = unix.MS_NOEXEC

	// SYNCHRONOUS will allow I/O to the file system to be done synchronously.
	SYNCHRONOUS = unix.MS_SYNCHRONOUS

	// DIRSYNC will force all directory updates within the file system to be done
	// synchronously. This affects the following system calls: create, link,
	// unlink, symlink, mkdir, rmdir, mknod and rename.
	DIRSYNC = unix.MS_DIRSYNC

	// REMOUNT will attempt to remount an already-mounted file system. This is
	// commonly used to change the mount flags for a file system, especially to
	// make a readonly file system writeable. It does not change device or mount
	// point.
	REMOUNT = unix.MS_REMOUNT

	// MANDLOCK will force mandatory locks on a filesystem.
	MANDLOCK = unix.MS_MANDLOCK

	// NOATIME will not update the file access time when reading from a file.
	NOATIME = unix.MS_NOATIME

	// NODIRATIME will not update the directory access time.
	NODIRATIME = unix.MS_NODIRATIME

	// BIND remounts a subtree somewhere else.
	BIND = unix.MS_BIND

	// RBIND remounts a subtree and all possible submounts somewhere else.
	RBIND = unix.MS_BIND | unix.MS_REC

	// UNBINDABLE creates a mount which cannot be cloned through a bind operation.
	UNBINDABLE = unix.MS_UNBINDABLE

	// RUNBINDABLE marks the entire mount tree as UNBINDABLE.
	RUNBINDABLE = unix.MS_UNBINDABLE | unix.MS_REC

	// PRIVATE creates a mount which carries no propagation abilities.
	PRIVATE = unix.MS_PRIVATE

	// RPRIVATE marks the entire mount tree as PRIVATE.
	RPRIVATE = unix.MS_PRIVATE | unix.MS_REC

	// SLAVE creates a mount which receives propagation from its master, but not
	// vice versa.
	SLAVE = unix.MS_SLAVE

	// RSLAVE marks the entire mount tree as SLAVE.
	RSLAVE = unix.MS_SLAVE | unix.MS_REC

	// SHARED creates a mount which provides the ability to create mirrors of
	// that mount such that mounts and unmounts within any of the mirrors
	// propagate to the other mirrors.
	SHARED = unix.MS_SHARED

	// RSHARED marks the entire mount tree as SHARED.
	RSHARED = unix.MS_SHARED | unix.MS_REC

	// RELATIME updates inode access times relative to modify or change time.
	RELATIME = unix.MS_RELATIME

	// STRICTATIME allows to explicitly request full atime updates.  This makes
	// it possible for the kernel to default to relatime or noatime but still
	// allow userspace to override it.
	STRICTATIME = unix.MS_STRICTATIME

	mntDetach = unix.MNT_DETACH
)
