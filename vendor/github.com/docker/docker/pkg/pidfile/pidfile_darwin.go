// +build darwin

package pidfile

import (
	"golang.org/x/sys/unix"
)

func processExists(pid int) bool {
	// OS X does not have a proc filesystem.
	// Use kill -0 pid to judge if the process exists.
	err := unix.Kill(pid, 0)
	return err == nil
}
