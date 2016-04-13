// +build linux

package reexec

import (
	"os/exec"
	"syscall"
)

// Command returns *exec.Cmd which have Path as current binary. Also it setting
// SysProcAttr.Pdeathsig to SIGTERM.
// For example if current binary is "docker" at "/usr/bin", then cmd.Path will
// be set to "/usr/bin/docker".
func Command(args ...string) *exec.Cmd {
	return &exec.Cmd{
		Path: Self(),
		Args: args,
		SysProcAttr: &syscall.SysProcAttr{
			Pdeathsig: syscall.SIGTERM,
		},
	}
}
