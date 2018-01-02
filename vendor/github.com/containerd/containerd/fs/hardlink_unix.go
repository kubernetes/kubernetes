// +build !windows

package fs

import (
	"os"
	"syscall"
)

func getLinkInfo(fi os.FileInfo) (uint64, bool) {
	s, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, false
	}

	return uint64(s.Ino), !fi.IsDir() && s.Nlink > 1
}
