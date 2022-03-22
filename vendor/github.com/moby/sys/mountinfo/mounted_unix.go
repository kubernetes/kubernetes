//go:build linux || (freebsd && cgo) || (openbsd && cgo) || (darwin && cgo)
// +build linux freebsd,cgo openbsd,cgo darwin,cgo

package mountinfo

import (
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

func mountedByStat(path string) (bool, error) {
	var st unix.Stat_t

	if err := unix.Lstat(path, &st); err != nil {
		return false, &os.PathError{Op: "stat", Path: path, Err: err}
	}
	dev := st.Dev
	parent := filepath.Dir(path)
	if err := unix.Lstat(parent, &st); err != nil {
		return false, &os.PathError{Op: "stat", Path: parent, Err: err}
	}
	if dev != st.Dev {
		// Device differs from that of parent,
		// so definitely a mount point.
		return true, nil
	}
	// NB: this does not detect bind mounts on Linux.
	return false, nil
}

func normalizePath(path string) (realPath string, err error) {
	if realPath, err = filepath.Abs(path); err != nil {
		return "", fmt.Errorf("unable to get absolute path for %q: %w", path, err)
	}
	if realPath, err = filepath.EvalSymlinks(realPath); err != nil {
		return "", fmt.Errorf("failed to canonicalise path for %q: %w", path, err)
	}
	if _, err := os.Stat(realPath); err != nil {
		return "", fmt.Errorf("failed to stat target of %q: %w", path, err)
	}
	return realPath, nil
}

func mountedByMountinfo(path string) (bool, error) {
	entries, err := GetMounts(SingleEntryFilter(path))
	if err != nil {
		return false, err
	}

	return len(entries) > 0, nil
}
