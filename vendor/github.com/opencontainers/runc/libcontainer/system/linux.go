//go:build linux
// +build linux

package system

import (
	"os"
	"os/exec"
	"runtime"
	"strings"
	"unsafe"

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

// Deprecated: Execv is not used in runc anymore, it will be removed in v1.2.0.
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
		if err != unix.EINTR { //nolint:errorlint // unix errors are bare
			return &os.PathError{Op: "exec", Path: cmd, Err: err}
		}
	}
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

func prepareAt(dir *os.File, path string) (int, string) {
	if dir == nil {
		return unix.AT_FDCWD, path
	}

	// Rather than just filepath.Join-ing path here, do it manually so the
	// error and handle correctly indicate cases like path=".." as being
	// relative to the correct directory. The handle.Name() might end up being
	// wrong but because this is (currently) only used in MkdirAllInRoot, that
	// isn't a problem.
	dirName := dir.Name()
	if !strings.HasSuffix(dirName, "/") {
		dirName += "/"
	}
	fullPath := dirName + path

	return int(dir.Fd()), fullPath
}

func Openat(dir *os.File, path string, flags int, mode uint32) (*os.File, error) {
	dirFd, fullPath := prepareAt(dir, path)
	fd, err := unix.Openat(dirFd, path, flags, mode)
	if err != nil {
		return nil, &os.PathError{Op: "openat", Path: fullPath, Err: err}
	}
	runtime.KeepAlive(dir)
	return os.NewFile(uintptr(fd), fullPath), nil
}

func Mkdirat(dir *os.File, path string, mode uint32) error {
	dirFd, fullPath := prepareAt(dir, path)
	err := unix.Mkdirat(dirFd, path, mode)
	if err != nil {
		err = &os.PathError{Op: "mkdirat", Path: fullPath, Err: err}
	}
	runtime.KeepAlive(dir)
	return err
}
