// +build freebsd,cgo

package mount

/*
#include <sys/mount.h>
*/
import "C"

const (
	RDONLY      = C.MNT_RDONLY
	NOSUID      = C.MNT_NOSUID
	NOEXEC      = C.MNT_NOEXEC
	SYNCHRONOUS = C.MNT_SYNCHRONOUS
	NOATIME     = C.MNT_NOATIME

	BIND        = 0
	DIRSYNC     = 0
	MANDLOCK    = 0
	NODEV       = 0
	NODIRATIME  = 0
	UNBINDABLE  = 0
	RUNBINDABLE = 0
	PRIVATE     = 0
	RPRIVATE    = 0
	SHARED      = 0
	RSHARED     = 0
	SLAVE       = 0
	RSLAVE      = 0
	RBIND       = 0
	RELATIVE    = 0
	RELATIME    = 0
	REMOUNT     = 0
	STRICTATIME = 0
)
