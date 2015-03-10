// +build linux

package console

import (
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"unsafe"

	"github.com/docker/libcontainer/label"
)

// Setup initializes the proper /dev/console inside the rootfs path
func Setup(rootfs, consolePath, mountLabel string) error {
	oldMask := syscall.Umask(0000)
	defer syscall.Umask(oldMask)

	if err := os.Chmod(consolePath, 0600); err != nil {
		return err
	}

	if err := os.Chown(consolePath, 0, 0); err != nil {
		return err
	}

	if err := label.SetFileLabel(consolePath, mountLabel); err != nil {
		return fmt.Errorf("set file label %s %s", consolePath, err)
	}

	dest := filepath.Join(rootfs, "dev/console")

	f, err := os.Create(dest)
	if err != nil && !os.IsExist(err) {
		return fmt.Errorf("create %s %s", dest, err)
	}

	if f != nil {
		f.Close()
	}

	if err := syscall.Mount(consolePath, dest, "bind", syscall.MS_BIND, ""); err != nil {
		return fmt.Errorf("bind %s to %s %s", consolePath, dest, err)
	}

	return nil
}

func OpenAndDup(consolePath string) error {
	slave, err := OpenTerminal(consolePath, syscall.O_RDWR)
	if err != nil {
		return fmt.Errorf("open terminal %s", err)
	}

	if err := syscall.Dup2(int(slave.Fd()), 0); err != nil {
		return err
	}

	if err := syscall.Dup2(int(slave.Fd()), 1); err != nil {
		return err
	}

	return syscall.Dup2(int(slave.Fd()), 2)
}

// Unlockpt unlocks the slave pseudoterminal device corresponding to the master pseudoterminal referred to by f.
// Unlockpt should be called before opening the slave side of a pseudoterminal.
func Unlockpt(f *os.File) error {
	var u int32

	return Ioctl(f.Fd(), syscall.TIOCSPTLCK, uintptr(unsafe.Pointer(&u)))
}

// Ptsname retrieves the name of the first available pts for the given master.
func Ptsname(f *os.File) (string, error) {
	var n int32

	if err := Ioctl(f.Fd(), syscall.TIOCGPTN, uintptr(unsafe.Pointer(&n))); err != nil {
		return "", err
	}

	return fmt.Sprintf("/dev/pts/%d", n), nil
}

// CreateMasterAndConsole will open /dev/ptmx on the host and retreive the
// pts name for use as the pty slave inside the container
func CreateMasterAndConsole() (*os.File, string, error) {
	master, err := os.OpenFile("/dev/ptmx", syscall.O_RDWR|syscall.O_NOCTTY|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, "", err
	}

	console, err := Ptsname(master)
	if err != nil {
		return nil, "", err
	}

	if err := Unlockpt(master); err != nil {
		return nil, "", err
	}

	return master, console, nil
}

// OpenPtmx opens /dev/ptmx, i.e. the PTY master.
func OpenPtmx() (*os.File, error) {
	// O_NOCTTY and O_CLOEXEC are not present in os package so we use the syscall's one for all.
	return os.OpenFile("/dev/ptmx", syscall.O_RDONLY|syscall.O_NOCTTY|syscall.O_CLOEXEC, 0)
}

// OpenTerminal is a clone of os.OpenFile without the O_CLOEXEC
// used to open the pty slave inside the container namespace
func OpenTerminal(name string, flag int) (*os.File, error) {
	r, e := syscall.Open(name, flag, 0)
	if e != nil {
		return nil, &os.PathError{Op: "open", Path: name, Err: e}
	}
	return os.NewFile(uintptr(r), name), nil
}

func Ioctl(fd uintptr, flag, data uintptr) error {
	if _, _, err := syscall.Syscall(syscall.SYS_IOCTL, fd, flag, data); err != 0 {
		return err
	}

	return nil
}
