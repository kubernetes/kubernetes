// +build darwin dragonfly freebsd !android,linux netbsd openbsd solaris
// +build cgo

package sftp

import (
	"os"
	"syscall"
)

func fileStatFromInfoOs(fi os.FileInfo, flags *uint32, fileStat *FileStat) {
	if statt, ok := fi.Sys().(*syscall.Stat_t); ok {
		*flags |= ssh_FILEXFER_ATTR_UIDGID
		fileStat.UID = statt.Uid
		fileStat.GID = statt.Gid
	}
}
