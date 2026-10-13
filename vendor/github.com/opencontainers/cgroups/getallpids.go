package cgroups

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// GetAllPids returns all pids from the cgroup identified by path, and all its
// sub-cgroups.
func GetAllPids(path string) ([]int, error) {
	var pids []int
	err := filepath.WalkDir(path, func(p string, d fs.DirEntry, iErr error) error {
		if iErr != nil {
			// A descendant cgroup can be removed while we walk, ignore
			// any such error unless it's on the root (path) cgroup here
			if p != path && ignoreCgroupRemoved(iErr) {
				return nil
			}
			return iErr
		}
		if !d.IsDir() {
			return nil
		}
		cPids, err := readProcsFile(p)
		if err != nil {
			if p != path && ignoreCgroupRemoved(err) {
				return nil
			}
			return err
		}
		pids = append(pids, cPids...)
		return nil
	})
	return pids, err
}

// ignoreCgroupRemoved reports whether err indicates the cgroup was removed.
func ignoreCgroupRemoved(err error) bool {
	return os.IsNotExist(err) || errors.Is(err, unix.ENODEV)
}
