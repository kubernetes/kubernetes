// +build windows

package reexec

import (
	"os/exec"
)

// Command returns *exec.Cmd which have Path as current binary.
// For example if current binary is "docker.exe" at "C:\", then cmd.Path will
// be set to "C:\docker.exe".
func Command(args ...string) *exec.Cmd {
	return &exec.Cmd{
		Path: Self(),
		Args: args,
	}
}
