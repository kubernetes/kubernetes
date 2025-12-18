//go:build !linux && !darwin
// +build !linux,!darwin

package pty

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
)

const Supported = false

func SetCtty(cmd *exec.Cmd, tty *os.File) error {
	panic("SetCtty called on unsupported platform")
}

func Open() (pty, tty *os.File, err error) {
	return nil, nil, fmt.Errorf("pty unsupported on %s", runtime.GOOS)
}
