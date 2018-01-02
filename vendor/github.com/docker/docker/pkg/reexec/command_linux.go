// +build linux

package reexec

import (
	"os/exec"
	"syscall"

	"golang.org/x/sys/unix"
)

// Self returns the path to the current process's binary.
// Returns "/proc/self/exe".
func Self() string {
	return "/proc/self/exe"
}

// Command returns *exec.Cmd which has Path as current binary. Also it setting
// SysProcAttr.Pdeathsig to SIGTERM.
// This will use the in-memory version (/proc/self/exe) of the current binary,
// it is thus safe to delete or replace the on-disk binary (os.Args[0]).
func Command(args ...string) *exec.Cmd {
	return &exec.Cmd{
		Path: Self(),
		Args: args,
		SysProcAttr: &syscall.SysProcAttr{
			Pdeathsig: unix.SIGTERM,
		},
	}
}
