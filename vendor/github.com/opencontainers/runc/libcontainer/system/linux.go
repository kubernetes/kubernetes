// +build linux

package system

import (
	"os/exec"
	"syscall"
	"unsafe"
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

	return syscall.Exec(name, args, env)
}

func SetParentDeathSignal(sig uintptr) error {
	if _, _, err := syscall.RawSyscall(syscall.SYS_PRCTL, syscall.PR_SET_PDEATHSIG, sig, 0); err != 0 {
		return err
	}
	return nil
}

func GetParentDeathSignal() (ParentDeathSignal, error) {
	var sig int
	_, _, err := syscall.RawSyscall(syscall.SYS_PRCTL, syscall.PR_GET_PDEATHSIG, uintptr(unsafe.Pointer(&sig)), 0)
	if err != 0 {
		return -1, err
	}
	return ParentDeathSignal(sig), nil
}

func SetKeepCaps() error {
	if _, _, err := syscall.RawSyscall(syscall.SYS_PRCTL, syscall.PR_SET_KEEPCAPS, 1, 0); err != 0 {
		return err
	}

	return nil
}

func ClearKeepCaps() error {
	if _, _, err := syscall.RawSyscall(syscall.SYS_PRCTL, syscall.PR_SET_KEEPCAPS, 0, 0); err != 0 {
		return err
	}

	return nil
}

func Setctty() error {
	if _, _, err := syscall.RawSyscall(syscall.SYS_IOCTL, 0, uintptr(syscall.TIOCSCTTY), 0); err != 0 {
		return err
	}
	return nil
}
