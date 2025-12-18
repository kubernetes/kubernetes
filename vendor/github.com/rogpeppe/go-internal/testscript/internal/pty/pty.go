//go:build linux || darwin
// +build linux darwin

package pty

import (
	"fmt"
	"os"
	"os/exec"
	"syscall"
)

const Supported = true

func SetCtty(cmd *exec.Cmd, tty *os.File) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setctty: true,
		Setsid:  true,
		Ctty:    3,
	}
	cmd.ExtraFiles = []*os.File{tty}
}

func Open() (pty, tty *os.File, err error) {
	p, err := os.OpenFile("/dev/ptmx", os.O_RDWR, 0)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open pty multiplexer: %v", err)
	}
	defer func() {
		if err != nil {
			p.Close()
		}
	}()

	name, err := ptyName(p)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to obtain tty name: %v", err)
	}

	if err := ptyGrant(p); err != nil {
		return nil, nil, fmt.Errorf("failed to grant pty: %v", err)
	}

	if err := ptyUnlock(p); err != nil {
		return nil, nil, fmt.Errorf("failed to unlock pty: %v", err)
	}

	t, err := os.OpenFile(name, os.O_RDWR|syscall.O_NOCTTY, 0)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open TTY: %v", err)
	}

	return p, t, nil
}

func ioctl(f *os.File, name string, cmd, ptr uintptr) error {
	_, _, err := syscall.Syscall(syscall.SYS_IOCTL, f.Fd(), cmd, ptr)
	if err != 0 {
		return fmt.Errorf("%s ioctl failed: %v", name, err)
	}
	return nil
}
