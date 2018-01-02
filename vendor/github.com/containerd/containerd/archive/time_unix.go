// +build linux freebsd solaris

package archive

import (
	"time"

	"golang.org/x/sys/unix"

	"github.com/pkg/errors"
)

func chtimes(path string, atime, mtime time.Time) error {
	var utimes [2]unix.Timespec
	utimes[0] = unix.NsecToTimespec(atime.UnixNano())
	utimes[1] = unix.NsecToTimespec(mtime.UnixNano())

	if err := unix.UtimesNanoAt(unix.AT_FDCWD, path, utimes[0:], unix.AT_SYMLINK_NOFOLLOW); err != nil {
		return errors.Wrap(err, "failed call to UtimesNanoAt")
	}

	return nil
}
