//go:build !windows
// +build !windows

package utils

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	_ "unsafe" // for go:linkname

	"github.com/opencontainers/runc/libcontainer/system"

	securejoin "github.com/cyphar/filepath-securejoin"
	"golang.org/x/sys/unix"
)

// EnsureProcHandle returns whether or not the given file handle is on procfs.
func EnsureProcHandle(fh *os.File) error {
	var buf unix.Statfs_t
	if err := unix.Fstatfs(int(fh.Fd()), &buf); err != nil {
		return fmt.Errorf("ensure %s is on procfs: %w", fh.Name(), err)
	}
	if buf.Type != unix.PROC_SUPER_MAGIC {
		return fmt.Errorf("%s is not on procfs", fh.Name())
	}
	return nil
}

type fdFunc func(fd int)

// fdRangeFrom calls the passed fdFunc for each file descriptor that is open in
// the current process.
func fdRangeFrom(minFd int, fn fdFunc) error {
	fdDir, err := os.Open("/proc/self/fd")
	if err != nil {
		return err
	}
	defer fdDir.Close()

	if err := EnsureProcHandle(fdDir); err != nil {
		return err
	}

	fdList, err := fdDir.Readdirnames(-1)
	if err != nil {
		return err
	}
	for _, fdStr := range fdList {
		fd, err := strconv.Atoi(fdStr)
		// Ignore non-numeric file names.
		if err != nil {
			continue
		}
		// Ignore descriptors lower than our specified minimum.
		if fd < minFd {
			continue
		}
		// Ignore the file descriptor we used for readdir, as it will be closed
		// when we return.
		if uintptr(fd) == fdDir.Fd() {
			continue
		}
		// Run the closure.
		fn(fd)
	}
	return nil
}

// CloseExecFrom sets the O_CLOEXEC flag on all file descriptors greater or
// equal to minFd in the current process.
func CloseExecFrom(minFd int) error {
	return fdRangeFrom(minFd, unix.CloseOnExec)
}

//go:linkname runtime_IsPollDescriptor internal/poll.IsPollDescriptor

// In order to make sure we do not close the internal epoll descriptors the Go
// runtime uses, we need to ensure that we skip descriptors that match
// "internal/poll".IsPollDescriptor. Yes, this is a Go runtime internal thing,
// unfortunately there's no other way to be sure we're only keeping the file
// descriptors the Go runtime needs. Hopefully nothing blows up doing this...
func runtime_IsPollDescriptor(fd uintptr) bool //nolint:revive

// UnsafeCloseFrom closes all file descriptors greater or equal to minFd in the
// current process, except for those critical to Go's runtime (such as the
// netpoll management descriptors).
//
// NOTE: That this function is incredibly dangerous to use in most Go code, as
// closing file descriptors from underneath *os.File handles can lead to very
// bad behaviour (the closed file descriptor can be re-used and then any
// *os.File operations would apply to the wrong file). This function is only
// intended to be called from the last stage of runc init.
func UnsafeCloseFrom(minFd int) error {
	// We must not close some file descriptors.
	return fdRangeFrom(minFd, func(fd int) {
		if runtime_IsPollDescriptor(uintptr(fd)) {
			// These are the Go runtimes internal netpoll file descriptors.
			// These file descriptors are operated on deep in the Go scheduler,
			// and closing those files from underneath Go can result in panics.
			// There is no issue with keeping them because they are not
			// executable and are not useful to an attacker anyway. Also we
			// don't have any choice.
			return
		}
		// There's nothing we can do about errors from close(2), and the
		// only likely error to be seen is EBADF which indicates the fd was
		// already closed (in which case, we got what we wanted).
		_ = unix.Close(fd)
	})
}

// NewSockPair returns a new unix socket pair
func NewSockPair(name string) (parent *os.File, child *os.File, err error) {
	fds, err := unix.Socketpair(unix.AF_LOCAL, unix.SOCK_STREAM|unix.SOCK_CLOEXEC, 0)
	if err != nil {
		return nil, nil, err
	}
	return os.NewFile(uintptr(fds[1]), name+"-p"), os.NewFile(uintptr(fds[0]), name+"-c"), nil
}

// IsLexicallyInRoot is shorthand for strings.HasPrefix(path+"/", root+"/"),
// but properly handling the case where path or root are "/".
//
// NOTE: The return value only make sense if the path doesn't contain "..".
func IsLexicallyInRoot(root, path string) bool {
	if root != "/" {
		root += "/"
	}
	if path != "/" {
		path += "/"
	}
	return strings.HasPrefix(path, root)
}

// MkdirAllInRootOpen attempts to make
//
//	path, _ := securejoin.SecureJoin(root, unsafePath)
//	os.MkdirAll(path, mode)
//	os.Open(path)
//
// safer against attacks where components in the path are changed between
// SecureJoin returning and MkdirAll (or Open) being called. In particular, we
// try to detect any symlink components in the path while we are doing the
// MkdirAll.
//
// NOTE: Unlike os.MkdirAll, mode is not Go's os.FileMode, it is the unix mode
// (the suid/sgid/sticky bits are not the same as for os.FileMode).
//
// NOTE: If unsafePath is a subpath of root, we assume that you have already
// called SecureJoin and so we use the provided path verbatim without resolving
// any symlinks (this is done in a way that avoids symlink-exchange races).
// This means that the path also must not contain ".." elements, otherwise an
// error will occur.
//
// This is a somewhat less safe alternative to
// <https://github.com/cyphar/filepath-securejoin/pull/13>, but it should
// detect attempts to trick us into creating directories outside of the root.
// We should migrate to securejoin.MkdirAll once it is merged.
func MkdirAllInRootOpen(root, unsafePath string, mode uint32) (_ *os.File, Err error) {
	// If the path is already "within" the root, use it verbatim.
	fullPath := unsafePath
	if !IsLexicallyInRoot(root, unsafePath) {
		var err error
		fullPath, err = securejoin.SecureJoin(root, unsafePath)
		if err != nil {
			return nil, err
		}
	}
	subPath, err := filepath.Rel(root, fullPath)
	if err != nil {
		return nil, err
	}

	// Check for any silly mode bits.
	if mode&^0o7777 != 0 {
		return nil, fmt.Errorf("tried to include non-mode bits in MkdirAll mode: 0o%.3o", mode)
	}

	currentDir, err := os.OpenFile(root, unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("open root handle: %w", err)
	}
	defer func() {
		if Err != nil {
			currentDir.Close()
		}
	}()

	for _, part := range strings.Split(subPath, string(filepath.Separator)) {
		switch part {
		case "", ".":
			// Skip over no-op components.
			continue
		case "..":
			return nil, fmt.Errorf("possible breakout detected: found %q component in SecureJoin subpath %s", part, subPath)
		}

		nextDir, err := system.Openat(currentDir, part, unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
		switch {
		case err == nil:
			// Update the currentDir.
			_ = currentDir.Close()
			currentDir = nextDir

		case errors.Is(err, unix.ENOTDIR):
			// This might be a symlink or some other random file. Either way,
			// error out.
			return nil, fmt.Errorf("cannot mkdir in %s/%s: %w", currentDir.Name(), part, unix.ENOTDIR)

		case errors.Is(err, os.ErrNotExist):
			// Luckily, mkdirat will not follow trailing symlinks, so this is
			// safe to do as-is.
			if err := system.Mkdirat(currentDir, part, mode); err != nil {
				return nil, err
			}
			// Open the new directory. There is a race here where an attacker
			// could swap the directory with a different directory, but
			// MkdirAll's fuzzy semantics mean we don't care about that.
			nextDir, err := system.Openat(currentDir, part, unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
			if err != nil {
				return nil, fmt.Errorf("open newly created directory: %w", err)
			}
			// Update the currentDir.
			_ = currentDir.Close()
			currentDir = nextDir

		default:
			return nil, err
		}
	}
	return currentDir, nil
}

// MkdirAllInRoot is a wrapper around MkdirAllInRootOpen which closes the
// returned handle, for callers that don't need to use it.
func MkdirAllInRoot(root, unsafePath string, mode uint32) error {
	f, err := MkdirAllInRootOpen(root, unsafePath, mode)
	if err == nil {
		_ = f.Close()
	}
	return err
}
