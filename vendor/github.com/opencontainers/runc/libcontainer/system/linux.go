//go:build linux
// +build linux

package system

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"syscall"
	"unsafe"

	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

type ParentDeathSignal int

func (p ParentDeathSignal) Restore() error {
	if p == 0 {
		return nil
	}
	current, err := GetParentDeathSignal()
	if err != nil {
		return err
	}
	if p == current {
		return nil
	}
	return p.Set()
}

func (p ParentDeathSignal) Set() error {
	return SetParentDeathSignal(uintptr(p))
}

func Execv(cmd string, args []string, env []string) error {
	name, err := exec.LookPath(cmd)
	if err != nil {
		return err
	}
	return Exec(name, args, env)
}

func Exec(cmd string, args []string, env []string) error {
	for {
		err := unix.Exec(cmd, args, env)
		if err != unix.EINTR {
			return &os.PathError{Op: "exec", Path: cmd, Err: err}
		}
	}
}

func execveat(fd uintptr, pathname string, args []string, env []string, flags int) error {
	pathnamep, err := syscall.BytePtrFromString(pathname)
	if err != nil {
		return err
	}

	argvp, err := syscall.SlicePtrFromStrings(args)
	if err != nil {
		return err
	}

	envp, err := syscall.SlicePtrFromStrings(env)
	if err != nil {
		return err
	}

	_, _, errno := syscall.Syscall6(
		unix.SYS_EXECVEAT,
		fd,
		uintptr(unsafe.Pointer(pathnamep)),
		uintptr(unsafe.Pointer(&argvp[0])),
		uintptr(unsafe.Pointer(&envp[0])),
		uintptr(flags),
		0,
	)
	return errno
}

func Fexecve(fd uintptr, args []string, env []string) error {
	var err error
	for {
		err = execveat(fd, "", args, env, unix.AT_EMPTY_PATH)
		if err != unix.EINTR { // nolint:errorlint // unix errors are bare
			break
		}
	}
	if err == unix.ENOSYS { // nolint:errorlint // unix errors are bare
		// Fallback to classic /proc/self/fd/... exec.
		return Exec("/proc/self/fd/"+strconv.Itoa(int(fd)), args, env)
	}
	return os.NewSyscallError("execveat", err)
}

func SetParentDeathSignal(sig uintptr) error {
	if err := unix.Prctl(unix.PR_SET_PDEATHSIG, sig, 0, 0, 0); err != nil {
		return err
	}
	return nil
}

func GetParentDeathSignal() (ParentDeathSignal, error) {
	var sig int
	if err := unix.Prctl(unix.PR_GET_PDEATHSIG, uintptr(unsafe.Pointer(&sig)), 0, 0, 0); err != nil {
		return -1, err
	}
	return ParentDeathSignal(sig), nil
}

func SetKeepCaps() error {
	if err := unix.Prctl(unix.PR_SET_KEEPCAPS, 1, 0, 0, 0); err != nil {
		return err
	}

	return nil
}

func ClearKeepCaps() error {
	if err := unix.Prctl(unix.PR_SET_KEEPCAPS, 0, 0, 0, 0); err != nil {
		return err
	}

	return nil
}

func Setctty() error {
	if err := unix.IoctlSetInt(0, unix.TIOCSCTTY, 0); err != nil {
		return err
	}
	return nil
}

// SetSubreaper sets the value i as the subreaper setting for the calling process
func SetSubreaper(i int) error {
	return unix.Prctl(unix.PR_SET_CHILD_SUBREAPER, uintptr(i), 0, 0, 0)
}

// GetSubreaper returns the subreaper setting for the calling process
func GetSubreaper() (int, error) {
	var i uintptr

	if err := unix.Prctl(unix.PR_GET_CHILD_SUBREAPER, uintptr(unsafe.Pointer(&i)), 0, 0, 0); err != nil {
		return -1, err
	}

	return int(i), nil
}

func ExecutableMemfd(comment string, flags int) (*os.File, error) {
	// Try to use MFD_EXEC first. On pre-6.3 kernels we get -EINVAL for this
	// flag. On post-6.3 kernels, with vm.memfd_noexec=1 this ensures we get an
	// executable memfd. For vm.memfd_noexec=2 this is a bit more complicated.
	// The original vm.memfd_noexec=2 implementation incorrectly silently
	// allowed MFD_EXEC[1] -- this should be fixed in 6.6. On 6.6 and newer
	// kernels, we will get -EACCES if we try to use MFD_EXEC with
	// vm.memfd_noexec=2 (for 6.3-6.5, -EINVAL was the intended return value).
	//
	// The upshot is we only need to retry without MFD_EXEC on -EINVAL because
	// it just so happens that passing MFD_EXEC bypasses vm.memfd_noexec=2 on
	// kernels where -EINVAL is actually a security denial.
	memfd, err := unix.MemfdCreate(comment, flags|unix.MFD_EXEC)
	if err == unix.EINVAL {
		memfd, err = unix.MemfdCreate(comment, flags)
	}
	if err != nil {
		if err == unix.EACCES {
			logrus.Info("memfd_create(MFD_EXEC) failed, possibly due to vm.memfd_noexec=2 -- falling back to less secure O_TMPFILE")
		}
		err := os.NewSyscallError("memfd_create", err)
		return nil, fmt.Errorf("failed to create executable memfd: %w", err)
	}
	return os.NewFile(uintptr(memfd), "/memfd:"+comment), nil
}

// Copy is like io.Copy except it uses sendfile(2) if the source and sink are
// both (*os.File) as an optimisation to make copies faster.
func Copy(dst io.Writer, src io.Reader) (copied int64, err error) {
	dstFile, _ := dst.(*os.File)
	srcFile, _ := src.(*os.File)

	if dstFile != nil && srcFile != nil {
		fi, err := srcFile.Stat()
		if err != nil {
			goto fallback
		}
		size := fi.Size()
		for size > 0 {
			n, err := unix.Sendfile(int(dstFile.Fd()), int(srcFile.Fd()), nil, int(size))
			if n > 0 {
				size -= int64(n)
				copied += int64(n)
			}
			if err == unix.EINTR {
				continue
			}
			if err != nil {
				if copied == 0 {
					// If we haven't copied anything so far, we can safely just
					// fallback to io.Copy. We could always do the fallback but
					// it's safer to error out in the case of a partial copy
					// followed by an error (which should never happen).
					goto fallback
				}
				return copied, fmt.Errorf("partial sendfile copy: %w", err)
			}
		}
		return copied, nil
	}

fallback:
	return io.Copy(dst, src)
}

// SetLinuxPersonality sets the Linux execution personality. For more information see the personality syscall documentation.
// checkout getLinuxPersonalityFromStr() from libcontainer/specconv/spec_linux.go for type conversion.
func SetLinuxPersonality(personality int) error {
	_, _, errno := unix.Syscall(unix.SYS_PERSONALITY, uintptr(personality), 0, 0)
	if errno != 0 {
		return &os.SyscallError{Syscall: "set_personality", Err: errno}
	}
	return nil
}
