// +build !linux,!windows,!freebsd,!solaris,!darwin

package reexec

import (
	"os/exec"
)

// Command is unsupported on operating systems apart from Linux, Windows, Solaris and Darwin.
func Command(args ...string) *exec.Cmd {
	return nil
}
