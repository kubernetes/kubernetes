// +build !linux

package main

import (
	"fmt"
	"path/filepath"
	"syscall"
)

func checkMountStatfs(d string, readonly bool) error {
	// or....
	// os.Stat(path).Sys().(*syscall.Stat_t).Dev
	sfs1 := &syscall.Statfs_t{}
	if err := syscall.Statfs(d, sfs1); err != nil {
		return fmt.Errorf("error calling statfs on %q: %v", d, err)
	}
	sfs2 := &syscall.Statfs_t{}
	if err := syscall.Statfs(filepath.Dir(d), sfs2); err != nil {
		return fmt.Errorf("error calling statfs on %q: %v", d, err)
	}
	if isSameFilesystem(sfs1, sfs2) {
		return fmt.Errorf("%q is not a mount point", d)
	}
	ro := sfs1.Flags&syscall.O_RDONLY == 1
	if ro != readonly {
		return fmt.Errorf("%q mounted ro=%t, want %t", d, ro, readonly)
	}

	return nil
}

func checkMountImpl(d string, readonly bool) error {
	return checkMountStatfs(d, readonly)
}
