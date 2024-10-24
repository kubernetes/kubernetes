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

	"github.com/opencontainers/runc/libcontainer/utils"
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
	if _, err := fd.WriteString(data); err != nil {
		// Having data in the error message helps in debugging.
		return fmt.Errorf("failed to write %q: %w", data, err)
	}
	return nil
}

// WriteFileByLine is the same as WriteFile, except if data contains newlines,
// it is written line by line.
func WriteFileByLine(dir, file, data string) error {
	i := strings.Index(data, "\n")
	if i == -1 {
		return WriteFile(dir, file, data)
	}

	fd, err := OpenFile(dir, file, unix.O_WRONLY)
	if err != nil {
		return err
	}
	defer fd.Close()
	start := 0
	for {
		var line string
		if i == -1 {
			line = data[start:]
		} else {
			line = data[start : start+i+1]
		}
		_, err := fd.WriteString(line)
		if err != nil {
			return fmt.Errorf("failed to write %q: %w", line, err)
		}
		if i == -1 {
			break
		}
		start += i + 1
		i = strings.Index(data[start:], "\n")
	}
	return nil
}

const (
	cgroupfsDir    = "/sys/fs/cgroup"
	cgroupfsPrefix = cgroupfsDir + "/"
)

var (
	// TestMode is set to true by unit tests that need "fake" cgroupfs.
	TestMode bool

	cgroupRootHandle *os.File
	prepOnce         sync.Once
	prepErr          error
	resolveFlags     uint64
)

func prepareOpenat2() error {
	prepOnce.Do(func() {
		fd, err := unix.Openat2(-1, cgroupfsDir, &unix.OpenHow{
			Flags: unix.O_DIRECTORY | unix.O_PATH | unix.O_CLOEXEC,
		})
		if err != nil {
			prepErr = &os.PathError{Op: "openat2", Path: cgroupfsDir, Err: err}
			if err != unix.ENOSYS {
				logrus.Warnf("falling back to securejoin: %s", prepErr)
			} else {
				logrus.Debug("openat2 not available, falling back to securejoin")
			}
			return
		}
		file := os.NewFile(uintptr(fd), cgroupfsDir)

		var st unix.Statfs_t
		if err := unix.Fstatfs(int(file.Fd()), &st); err != nil {
			prepErr = &os.PathError{Op: "statfs", Path: cgroupfsDir, Err: err}
			logrus.Warnf("falling back to securejoin: %s", prepErr)
			return
		}

		cgroupRootHandle = file
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
	path := path.Join(dir, utils.CleanPath(file))
	if prepareOpenat2() != nil {
		return openFallback(path, flags, mode)
	}
	relPath := strings.TrimPrefix(path, cgroupfsPrefix)
	if len(relPath) == len(path) { // non-standard path, old system?
		return openFallback(path, flags, mode)
	}

	fd, err := unix.Openat2(int(cgroupRootHandle.Fd()), relPath,
		&unix.OpenHow{
			Resolve: resolveFlags,
			Flags:   uint64(flags) | unix.O_CLOEXEC,
			Mode:    uint64(mode),
		})
	if err != nil {
		err = &os.PathError{Op: "openat2", Path: path, Err: err}
		// Check if cgroupRootHandle is still opened to cgroupfsDir
		// (happens when this package is incorrectly used
		// across the chroot/pivot_root/mntns boundary, or
		// when /sys/fs/cgroup is remounted).
		//
		// TODO: if such usage will ever be common, amend this
		// to reopen cgroupRootHandle and retry openat2.
		fdPath, closer := utils.ProcThreadSelf("fd/" + strconv.Itoa(int(cgroupRootHandle.Fd())))
		defer closer()
		fdDest, _ := os.Readlink(fdPath)
		if fdDest != cgroupfsDir {
			// Wrap the error so it is clear that cgroupRootHandle
			// is opened to an unexpected/wrong directory.
			err = fmt.Errorf("cgroupRootHandle %d unexpectedly opened to %s != %s: %w",
				cgroupRootHandle.Fd(), fdDest, cgroupfsDir, err)
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
