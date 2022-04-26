// +build darwin freebsd netbsd

package git

import (
	"syscall"
	"time"

	"github.com/go-git/go-git/v5/plumbing/format/index"
)

func init() {
	fillSystemInfo = func(e *index.Entry, sys interface{}) {
		if os, ok := sys.(*syscall.Stat_t); ok {
			e.CreatedAt = time.Unix(int64(os.Atimespec.Sec), int64(os.Atimespec.Nsec))
			e.Dev = uint32(os.Dev)
			e.Inode = uint32(os.Ino)
			e.GID = os.Gid
			e.UID = os.Uid
		}
	}
}

func isSymlinkWindowsNonAdmin(err error) bool {
	return false
}
