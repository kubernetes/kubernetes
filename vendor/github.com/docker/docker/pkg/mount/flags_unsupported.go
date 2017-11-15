// +build !linux,!freebsd freebsd,!cgo

package mount

// These flags are unsupported.
const (
	BIND        = 0
	DIRSYNC     = 0
	MANDLOCK    = 0
	NOATIME     = 0
	NODEV       = 0
	NODIRATIME  = 0
	NOEXEC      = 0
	NOSUID      = 0
	UNBINDABLE  = 0
	RUNBINDABLE = 0
	PRIVATE     = 0
	RPRIVATE    = 0
	SHARED      = 0
	RSHARED     = 0
	SLAVE       = 0
	RSLAVE      = 0
	RBIND       = 0
	RELATIME    = 0
	RELATIVE    = 0
	REMOUNT     = 0
	STRICTATIME = 0
	SYNCHRONOUS = 0
	RDONLY      = 0
	mntDetach   = 0
)
