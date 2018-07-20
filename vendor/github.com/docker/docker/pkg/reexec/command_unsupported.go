// +build !linux,!windows,!freebsd,!darwin

package reexec

import (
	"os/exec"
)

// Command is unsupported on operating systems apart from Linux, Windows, and Darwin.
func Command(args ...string) *exec.Cmd {
	return nil
}
