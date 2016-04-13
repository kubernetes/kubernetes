// +build !linux,!windows

package reexec

import (
	"os/exec"
)

// Command is unsupported on operating systems apart from Linux and Windows.
func Command(args ...string) *exec.Cmd {
	return nil
}
