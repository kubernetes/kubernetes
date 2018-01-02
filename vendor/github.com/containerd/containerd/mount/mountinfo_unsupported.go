// +build !linux,!freebsd,!solaris freebsd,!cgo solaris,!cgo

package mount

import (
	"fmt"
	"runtime"
)

// Self retrieves a list of mounts for the current running process.
func Self() ([]Info, error) {
	return nil, fmt.Errorf("mountinfo.Self is not implemented on %s/%s", runtime.GOOS, runtime.GOARCH)
}

// PID collects the mounts for a specific process ID.
func PID(pid int) ([]Info, error) {
	return nil, fmt.Errorf("mountinfo.PID is not implemented on %s/%s", runtime.GOOS, runtime.GOARCH)
}
