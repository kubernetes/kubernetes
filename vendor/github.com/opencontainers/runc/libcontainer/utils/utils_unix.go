//go:build !windows

package utils

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	_ "unsafe" // for go:linkname

	securejoin "github.com/cyphar/filepath-securejoin"
	"github.com/sirupsen/logrus"
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

var (
	haveCloseRangeCloexecBool bool
	haveCloseRangeCloexecOnce sync.Once
)

func haveCloseRangeCloexec() bool {
	haveCloseRangeCloexecOnce.Do(func() {
		// Make sure we're not closing a random file descriptor.
		tmpFd, err := unix.FcntlInt(0, unix.F_DUPFD_CLOEXEC, 0)
		if err != nil {
			return
		}
		defer unix.Close(tmpFd)

		err = unix.CloseRange(uint(tmpFd), uint(tmpFd), unix.CLOSE_RANGE_CLOEXEC)
		// Any error means we cannot use close_range(CLOSE_RANGE_CLOEXEC).
		// -ENOSYS and -EINVAL ultimately mean we don't have support, but any
		// other potential error would imply that even the most basic close
		// operation wouldn't work.
		haveCloseRangeCloexecBool = err == nil
	})
	return haveCloseRangeCloexecBool
}

type fdFunc func(fd int)

// fdRangeFrom calls the passed fdFunc for each file descriptor that is open in
// the current process.
func fdRangeFrom(minFd int, fn fdFunc) error {
	procSelfFd, closer := ProcThreadSelf("fd")
	defer closer()

	fdDir, err := os.Open(procSelfFd)
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
	// Use close_range(CLOSE_RANGE_CLOEXEC) if possible.
	if haveCloseRangeCloexec() {
		err := unix.CloseRange(uint(minFd), math.MaxUint, unix.CLOSE_RANGE_CLOEXEC)
		return os.NewSyscallError("close_range", err)
	}
	// Otherwise, fall back to the standard loop.
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
	// We cannot use close_range(2) even if it is available, because we must
	// not close some file descriptors.
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

// NewSockPair returns a new SOCK_STREAM unix socket pair.
func NewSockPair(name string) (parent, child *os.File, err error) {
	fds, err := unix.Socketpair(unix.AF_LOCAL, unix.SOCK_STREAM|unix.SOCK_CLOEXEC, 0)
	if err != nil {
		return nil, nil, err
	}
	return os.NewFile(uintptr(fds[1]), name+"-p"), os.NewFile(uintptr(fds[0]), name+"-c"), nil
}

// WithProcfd runs the passed closure with a procfd path (/proc/self/fd/...)
// corresponding to the unsafePath resolved within the root. Before passing the
// fd, this path is verified to have been inside the root -- so operating on it
// through the passed fdpath should be safe. Do not access this path through
// the original path strings, and do not attempt to use the pathname outside of
// the passed closure (the file handle will be freed once the closure returns).
func WithProcfd(root, unsafePath string, fn func(procfd string) error) error {
	// Remove the root then forcefully resolve inside the root.
	unsafePath = stripRoot(root, unsafePath)
	path, err := securejoin.SecureJoin(root, unsafePath)
	if err != nil {
		return fmt.Errorf("resolving path inside rootfs failed: %w", err)
	}

	procSelfFd, closer := ProcThreadSelf("fd/")
	defer closer()

	// Open the target path.
	fh, err := os.OpenFile(path, unix.O_PATH|unix.O_CLOEXEC, 0)
	if err != nil {
		return fmt.Errorf("open o_path procfd: %w", err)
	}
	defer fh.Close()

	procfd := filepath.Join(procSelfFd, strconv.Itoa(int(fh.Fd())))
	// Double-check the path is the one we expected.
	if realpath, err := os.Readlink(procfd); err != nil {
		return fmt.Errorf("procfd verification failed: %w", err)
	} else if realpath != path {
		return fmt.Errorf("possibly malicious path detected -- refusing to operate on %s", realpath)
	}

	return fn(procfd)
}

type ProcThreadSelfCloser func()

var (
	haveProcThreadSelf     bool
	haveProcThreadSelfOnce sync.Once
)

// ProcThreadSelf returns a string that is equivalent to
// /proc/thread-self/<subpath>, with a graceful fallback on older kernels where
// /proc/thread-self doesn't exist. This method DOES NOT use SecureJoin,
// meaning that the passed string needs to be trusted. The caller _must_ call
// the returned procThreadSelfCloser function (which is runtime.UnlockOSThread)
// *only once* after it has finished using the returned path string.
func ProcThreadSelf(subpath string) (string, ProcThreadSelfCloser) {
	haveProcThreadSelfOnce.Do(func() {
		if _, err := os.Stat("/proc/thread-self/"); err == nil {
			haveProcThreadSelf = true
		} else {
			logrus.Debugf("cannot stat /proc/thread-self (%v), falling back to /proc/self/task/<tid>", err)
		}
	})

	// We need to lock our thread until the caller is done with the path string
	// because any non-atomic operation on the path (such as opening a file,
	// then reading it) could be interrupted by the Go runtime where the
	// underlying thread is swapped out and the original thread is killed,
	// resulting in pull-your-hair-out-hard-to-debug issues in the caller. In
	// addition, the pre-3.17 fallback makes everything non-atomic because the
	// same thing could happen between unix.Gettid() and the path operations.
	//
	// In theory, we don't need to lock in the atomic user case when using
	// /proc/thread-self/, but it's better to be safe than sorry (and there are
	// only one or two truly atomic users of /proc/thread-self/).
	runtime.LockOSThread()

	threadSelf := "/proc/thread-self/"
	if !haveProcThreadSelf {
		// Pre-3.17 kernels did not have /proc/thread-self, so do it manually.
		threadSelf = "/proc/self/task/" + strconv.Itoa(unix.Gettid()) + "/"
		if _, err := os.Stat(threadSelf); err != nil {
			// Unfortunately, this code is called from rootfs_linux.go where we
			// are running inside the pid namespace of the container but /proc
			// is the host's procfs. Unfortunately there is no real way to get
			// the correct tid to use here (the kernel age means we cannot do
			// things like set up a private fsopen("proc") -- even scanning
			// NSpid in all of the tasks in /proc/self/task/*/status requires
			// Linux 4.1).
			//
			// So, we just have to assume that /proc/self is acceptable in this
			// one specific case.
			if os.Getpid() == 1 {
				logrus.Debugf("/proc/thread-self (tid=%d) cannot be emulated inside the initial container setup -- using /proc/self instead: %v", unix.Gettid(), err)
			} else {
				// This should never happen, but the fallback should work in most cases...
				logrus.Warnf("/proc/thread-self could not be emulated for pid=%d (tid=%d) -- using more buggy /proc/self fallback instead: %v", os.Getpid(), unix.Gettid(), err)
			}
			threadSelf = "/proc/self/"
		}
	}
	return threadSelf + subpath, runtime.UnlockOSThread
}

// ProcThreadSelfFd is small wrapper around ProcThreadSelf to make it easier to
// create a /proc/thread-self handle for given file descriptor.
//
// It is basically equivalent to ProcThreadSelf(fmt.Sprintf("fd/%d", fd)), but
// without using fmt.Sprintf to avoid unneeded overhead.
func ProcThreadSelfFd(fd uintptr) (string, ProcThreadSelfCloser) {
	return ProcThreadSelf("fd/" + strconv.FormatUint(uint64(fd), 10))
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
// This uses securejoin.MkdirAllHandle under the hood, but it has special
// handling if unsafePath has already been scoped within the rootfs (this is
// needed for a lot of runc callers and fixing this would require reworking a
// lot of path logic).
func MkdirAllInRootOpen(root, unsafePath string, mode uint32) (_ *os.File, Err error) {
	// If the path is already "within" the root, get the path relative to the
	// root and use that as the unsafe path. This is necessary because a lot of
	// MkdirAllInRootOpen callers have already done SecureJoin, and refactoring
	// all of them to stop using these SecureJoin'd paths would require a fair
	// amount of work.
	// TODO(cyphar): Do the refactor to libpathrs once it's ready.
	if IsLexicallyInRoot(root, unsafePath) {
		subPath, err := filepath.Rel(root, unsafePath)
		if err != nil {
			return nil, err
		}
		unsafePath = subPath
	}

	// Check for any silly mode bits.
	if mode&^0o7777 != 0 {
		return nil, fmt.Errorf("tried to include non-mode bits in MkdirAll mode: 0o%.3o", mode)
	}
	// Linux (and thus os.MkdirAll) silently ignores the suid and sgid bits if
	// passed. While it would make sense to return an error in that case (since
	// the user has asked for a mode that won't be applied), for compatibility
	// reasons we have to ignore these bits.
	if ignoredBits := mode &^ 0o1777; ignoredBits != 0 {
		logrus.Warnf("MkdirAll called with no-op mode bits that are ignored by Linux: 0o%.3o", ignoredBits)
		mode &= 0o1777
	}

	rootDir, err := os.OpenFile(root, unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("open root handle: %w", err)
	}
	defer rootDir.Close()

	return securejoin.MkdirAllHandle(rootDir, unsafePath, int(mode))
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

// Openat is a Go-friendly openat(2) wrapper.
func Openat(dir *os.File, path string, flags int, mode uint32) (*os.File, error) {
	dirFd := unix.AT_FDCWD
	if dir != nil {
		dirFd = int(dir.Fd())
	}
	flags |= unix.O_CLOEXEC

	fd, err := unix.Openat(dirFd, path, flags, mode)
	if err != nil {
		return nil, &os.PathError{Op: "openat", Path: path, Err: err}
	}
	return os.NewFile(uintptr(fd), dir.Name()+"/"+path), nil
}
