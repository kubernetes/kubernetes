package fscommon

import (
	"os"
	"strings"
	"sync"

	securejoin "github.com/cyphar/filepath-securejoin"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

const (
	cgroupfsDir    = "/sys/fs/cgroup"
	cgroupfsPrefix = cgroupfsDir + "/"
)

var (
	// Set to true by fs unit tests
	TestMode bool

	cgroupFd     int = -1
	prepOnce     sync.Once
	prepErr      error
	resolveFlags uint64
)

func prepareOpenat2() error {
	prepOnce.Do(func() {
		fd, err := unix.Openat2(-1, cgroupfsDir, &unix.OpenHow{
			Flags: unix.O_DIRECTORY | unix.O_PATH})
		if err != nil {
			prepErr = &os.PathError{Op: "openat2", Path: cgroupfsDir, Err: err}
			if err != unix.ENOSYS {
				logrus.Warnf("falling back to securejoin: %s", prepErr)
			} else {
				logrus.Debug("openat2 not available, falling back to securejoin")
			}
			return
		}
		var st unix.Statfs_t
		if err = unix.Fstatfs(fd, &st); err != nil {
			prepErr = &os.PathError{Op: "statfs", Path: cgroupfsDir, Err: err}
			logrus.Warnf("falling back to securejoin: %s", prepErr)
			return
		}

		cgroupFd = fd

		resolveFlags = unix.RESOLVE_BENEATH | unix.RESOLVE_NO_MAGICLINKS
		if st.Type == unix.CGROUP2_SUPER_MAGIC {
			// cgroupv2 has a single mountpoint and no "cpu,cpuacct" symlinks
			resolveFlags |= unix.RESOLVE_NO_XDEV | unix.RESOLVE_NO_SYMLINKS
		}

	})

	return prepErr
}

// OpenFile opens a cgroup file in a given dir with given flags.
// It is supposed to be used for cgroup files only.
func OpenFile(dir, file string, flags int) (*os.File, error) {
	if dir == "" {
		return nil, errors.Errorf("no directory specified for %s", file)
	}
	mode := os.FileMode(0)
	if TestMode && flags&os.O_WRONLY != 0 {
		// "emulate" cgroup fs for unit tests
		flags |= os.O_TRUNC | os.O_CREATE
		mode = 0o600
	}
	reldir := strings.TrimPrefix(dir, cgroupfsPrefix)
	if len(reldir) == len(dir) { // non-standard path, old system?
		return openWithSecureJoin(dir, file, flags, mode)
	}
	if prepareOpenat2() != nil {
		return openWithSecureJoin(dir, file, flags, mode)
	}

	relname := reldir + "/" + file
	fd, err := unix.Openat2(cgroupFd, relname,
		&unix.OpenHow{
			Resolve: resolveFlags,
			Flags:   uint64(flags) | unix.O_CLOEXEC,
			Mode:    uint64(mode),
		})
	if err != nil {
		return nil, &os.PathError{Op: "openat2", Path: dir + "/" + file, Err: err}
	}

	return os.NewFile(uintptr(fd), cgroupfsPrefix+relname), nil
}

func openWithSecureJoin(dir, file string, flags int, mode os.FileMode) (*os.File, error) {
	path, err := securejoin.SecureJoin(dir, file)
	if err != nil {
		return nil, err
	}

	return os.OpenFile(path, flags, mode)
}
