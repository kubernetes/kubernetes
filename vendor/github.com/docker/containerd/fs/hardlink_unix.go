// +build !windows

package fs

import (
	"errors"
	"os"
	"syscall"
)

func getHardLink(name string, fi os.FileInfo, inodes map[uint64]string) (string, error) {
	if fi.IsDir() {
		return "", nil
	}

	s, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return "", errors.New("unsupported stat type")
	}

	// If inode is not hardlinked, no reason to lookup or save inode
	if s.Nlink == 1 {
		return "", nil
	}

	inode := uint64(s.Ino)

	path, ok := inodes[inode]
	if !ok {
		inodes[inode] = name
	}
	return path, nil
}
