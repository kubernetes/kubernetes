// +build !windows

package mount // import "github.com/docker/docker/pkg/mount"

import "golang.org/x/sys/unix"

func unmount(target string, flags int) error {
	err := unix.Unmount(target, flags)
	if err == nil || err == unix.EINVAL {
		// Ignore "not mounted" error here. Note the same error
		// can be returned if flags are invalid, so this code
		// assumes that the flags value is always correct.
		return nil
	}

	return &mountError{
		op:     "umount",
		target: target,
		flags:  uintptr(flags),
		err:    err,
	}
}
