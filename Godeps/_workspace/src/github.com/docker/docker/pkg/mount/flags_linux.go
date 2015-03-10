package mount

import (
	"syscall"
)

const (
	RDONLY      = syscall.MS_RDONLY
	NOSUID      = syscall.MS_NOSUID
	NODEV       = syscall.MS_NODEV
	NOEXEC      = syscall.MS_NOEXEC
	SYNCHRONOUS = syscall.MS_SYNCHRONOUS
	DIRSYNC     = syscall.MS_DIRSYNC
	REMOUNT     = syscall.MS_REMOUNT
	MANDLOCK    = syscall.MS_MANDLOCK
	NOATIME     = syscall.MS_NOATIME
	NODIRATIME  = syscall.MS_NODIRATIME
	BIND        = syscall.MS_BIND
	RBIND       = syscall.MS_BIND | syscall.MS_REC
	UNBINDABLE  = syscall.MS_UNBINDABLE
	RUNBINDABLE = syscall.MS_UNBINDABLE | syscall.MS_REC
	PRIVATE     = syscall.MS_PRIVATE
	RPRIVATE    = syscall.MS_PRIVATE | syscall.MS_REC
	SLAVE       = syscall.MS_SLAVE
	RSLAVE      = syscall.MS_SLAVE | syscall.MS_REC
	SHARED      = syscall.MS_SHARED
	RSHARED     = syscall.MS_SHARED | syscall.MS_REC
	RELATIME    = syscall.MS_RELATIME
	STRICTATIME = syscall.MS_STRICTATIME
)
