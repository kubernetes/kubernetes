package mountinfo

import (
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// MountedFast is a method of detecting a mount point without reading
// mountinfo from procfs. A caller can only trust the result if no error
// and sure == true are returned. Otherwise, other methods (e.g. parsing
// /proc/mounts) have to be used. If unsure, use Mounted instead (which
// uses MountedFast, but falls back to parsing mountinfo if needed).
//
// If a non-existent path is specified, an appropriate error is returned.
// In case the caller is not interested in this particular error, it should
// be handled separately using e.g. errors.Is(err, fs.ErrNotExist).
//
// This function is only available on Linux. When available (since kernel
// v5.6), openat2(2) syscall is used to reliably detect all mounts. Otherwise,
// the implementation falls back to using stat(2), which can reliably detect
// normal (but not bind) mounts.
func MountedFast(path string) (mounted, sure bool, err error) {
	// Root is always mounted.
	if path == string(os.PathSeparator) {
		return true, true, nil
	}

	path, err = normalizePath(path)
	if err != nil {
		return false, false, err
	}
	mounted, sure, err = mountedFast(path)
	return
}

// mountedByOpenat2 is a method of detecting a mount that works for all kinds
// of mounts (incl. bind mounts), but requires a recent (v5.6+) linux kernel.
func mountedByOpenat2(path string) (bool, error) {
	dir, last := filepath.Split(path)

	dirfd, err := unix.Openat2(unix.AT_FDCWD, dir, &unix.OpenHow{
		Flags: unix.O_PATH | unix.O_CLOEXEC,
	})
	if err != nil {
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
	}
	// not sure
	return false, &os.PathError{Op: "openat2", Path: path, Err: err}
}

// mountedFast is similar to MountedFast, except it expects a normalized path.
func mountedFast(path string) (mounted, sure bool, err error) {
	// Root is always mounted.
	if path == string(os.PathSeparator) {
		return true, true, nil
	}

	// Try a fast path, using openat2() with RESOLVE_NO_XDEV.
	mounted, err = mountedByOpenat2(path)
	if err == nil {
		return mounted, true, nil
	}

	// Another fast path: compare st.st_dev fields.
	mounted, err = mountedByStat(path)
	// This does not work for bind mounts, so false negative
	// is possible, therefore only trust if return is true.
	if mounted && err == nil {
		return true, true, nil
	}

	return
}

func mounted(path string) (bool, error) {
	path, err := normalizePath(path)
	if err != nil {
		return false, err
	}
	mounted, sure, err := mountedFast(path)
	if sure && err == nil {
		return mounted, nil
	}

	// Fallback to parsing mountinfo.
	return mountedByMountinfo(path)
}
