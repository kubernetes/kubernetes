package mountinfo

import (
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// mountedByOpenat2 is a method of detecting a mount that works for all kinds
// of mounts (incl. bind mounts), but requires a recent (v5.6+) linux kernel.
func mountedByOpenat2(path string) (bool, error) {
	dir, last := filepath.Split(path)

	dirfd, err := unix.Openat2(unix.AT_FDCWD, dir, &unix.OpenHow{
		Flags: unix.O_PATH | unix.O_CLOEXEC,
	})
	if err != nil {
		if err == unix.ENOENT { // not a mount
			return false, nil
		}
		return false, &os.PathError{Op: "openat2", Path: dir, Err: err}
	}
	fd, err := unix.Openat2(dirfd, last, &unix.OpenHow{
		Flags:   unix.O_PATH | unix.O_CLOEXEC | unix.O_NOFOLLOW,
		Resolve: unix.RESOLVE_NO_XDEV,
	})
	_ = unix.Close(dirfd)
	switch err {
	case nil: // definitely not a mount
		_ = unix.Close(fd)
		return false, nil
	case unix.EXDEV: // definitely a mount
		return true, nil
	case unix.ENOENT: // not a mount
		return false, nil
	}
	// not sure
	return false, &os.PathError{Op: "openat2", Path: path, Err: err}
}

func mounted(path string) (bool, error) {
	// Try a fast path, using openat2() with RESOLVE_NO_XDEV.
	mounted, err := mountedByOpenat2(path)
	if err == nil {
		return mounted, nil
	}
	// Another fast path: compare st.st_dev fields.
	mounted, err = mountedByStat(path)
	// This does not work for bind mounts, so false negative
	// is possible, therefore only trust if return is true.
	if mounted && err == nil {
		return mounted, nil
	}

	// Fallback to parsing mountinfo
	return mountedByMountinfo(path)
}
