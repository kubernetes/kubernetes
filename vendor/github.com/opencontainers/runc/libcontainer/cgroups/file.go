package cgroups

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"

	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

// OpenFile opens a cgroup file in a given dir with given flags.
// It is supposed to be used for cgroup files only, and returns
// an error if the file is not a cgroup file.
//
// Arguments dir and file are joined together to form an absolute path
// to a file being opened.
func OpenFile(dir, file string, flags int) (*os.File, error) {
	if dir == "" {
		return nil, fmt.Errorf("no directory specified for %s", file)
	}
	return openFile(dir, file, flags)
}

// ReadFile reads data from a cgroup file in dir.
// It is supposed to be used for cgroup files only.
func ReadFile(dir, file string) (string, error) {
	fd, err := OpenFile(dir, file, unix.O_RDONLY)
	if err != nil {
		return "", err
	}
	defer fd.Close()
	var buf bytes.Buffer

	_, err = buf.ReadFrom(fd)
	return buf.String(), err
}

// WriteFile writes data to a cgroup file in dir.
// It is supposed to be used for cgroup files only.
func WriteFile(dir, file, data string) error {
	fd, err := OpenFile(dir, file, unix.O_WRONLY)
	if err != nil {
		return err
	}
	defer fd.Close()
	if err := retryingWriteFile(fd, data); err != nil {
		// Having data in the error message helps in debugging.
		return fmt.Errorf("failed to write %q: %w", data, err)
	}
	return nil
}

func retryingWriteFile(fd *os.File, data string) error {
	for {
		_, err := fd.Write([]byte(data))
		if errors.Is(err, unix.EINTR) {
			logrus.Infof("interrupted while writing %s to %s", data, fd.Name())
			continue
		}
		return err
	}
}

const (
	cgroupfsDir    = "/sys/fs/cgroup"
	cgroupfsPrefix = cgroupfsDir + "/"
)

var (
	// TestMode is set to true by unit tests that need "fake" cgroupfs.
	TestMode bool

	cgroupFd     int = -1
	prepOnce     sync.Once
	prepErr      error
	resolveFlags uint64
)

func prepareOpenat2() error {
	prepOnce.Do(func() {
		fd, err := unix.Openat2(-1, cgroupfsDir, &unix.OpenHow{
			Flags: unix.O_DIRECTORY | unix.O_PATH,
		})
		if err != nil {
			prepErr = &os.PathError{Op: "openat2", Path: cgroupfsDir, Err: err}
			if err != unix.ENOSYS { //nolint:errorlint // unix errors are bare
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

func openFile(dir, file string, flags int) (*os.File, error) {
	mode := os.FileMode(0)
	if TestMode && flags&os.O_WRONLY != 0 {
		// "emulate" cgroup fs for unit tests
		flags |= os.O_TRUNC | os.O_CREATE
		mode = 0o600
	}
	path := path.Join(dir, file)
	if prepareOpenat2() != nil {
		return openFallback(path, flags, mode)
	}
	relPath := strings.TrimPrefix(path, cgroupfsPrefix)
	if len(relPath) == len(path) { // non-standard path, old system?
		return openFallback(path, flags, mode)
	}

	fd, err := unix.Openat2(cgroupFd, relPath,
		&unix.OpenHow{
			Resolve: resolveFlags,
			Flags:   uint64(flags) | unix.O_CLOEXEC,
			Mode:    uint64(mode),
		})
	if err != nil {
		err = &os.PathError{Op: "openat2", Path: path, Err: err}
		// Check if cgroupFd is still opened to cgroupfsDir
		// (happens when this package is incorrectly used
		// across the chroot/pivot_root/mntns boundary, or
		// when /sys/fs/cgroup is remounted).
		//
		// TODO: if such usage will ever be common, amend this
		// to reopen cgroupFd and retry openat2.
		fdStr := strconv.Itoa(cgroupFd)
		fdDest, _ := os.Readlink("/proc/self/fd/" + fdStr)
		if fdDest != cgroupfsDir {
			// Wrap the error so it is clear that cgroupFd
			// is opened to an unexpected/wrong directory.
			err = fmt.Errorf("cgroupFd %s unexpectedly opened to %s != %s: %w",
				fdStr, fdDest, cgroupfsDir, err)
		}
		return nil, err
	}

	return os.NewFile(uintptr(fd), path), nil
}

var errNotCgroupfs = errors.New("not a cgroup file")

// Can be changed by unit tests.
var openFallback = openAndCheck

// openAndCheck is used when openat2(2) is not available. It checks the opened
// file is on cgroupfs, returning an error otherwise.
func openAndCheck(path string, flags int, mode os.FileMode) (*os.File, error) {
	fd, err := os.OpenFile(path, flags, mode)
	if err != nil {
		return nil, err
	}
	if TestMode {
		return fd, nil
	}
	// Check this is a cgroupfs file.
	var st unix.Statfs_t
	if err := unix.Fstatfs(int(fd.Fd()), &st); err != nil {
		_ = fd.Close()
		return nil, &os.PathError{Op: "statfs", Path: path, Err: err}
	}
	if st.Type != unix.CGROUP_SUPER_MAGIC && st.Type != unix.CGROUP2_SUPER_MAGIC {
		_ = fd.Close()
		return nil, &os.PathError{Op: "open", Path: path, Err: errNotCgroupfs}
	}

	return fd, nil
}
