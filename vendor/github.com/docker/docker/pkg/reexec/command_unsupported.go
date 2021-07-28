// +build !linux,!windows,!freebsd,!darwin

package reexec // import "github.com/docker/docker/pkg/reexec"

import (
	"os/exec"
)

func Self() string {
	return ""
}

// Command is unsupported on operating systems apart from Linux, Windows, and Darwin.
func Command(args ...string) *exec.Cmd {
	return nil
}
