// +build linux

package daemon

import (
	"strings"

	"github.com/docker/docker/daemon/execdriver/lxc"
)

// checkExecSupport returns an error if the exec driver does not support exec,
// or nil if it is supported.
func checkExecSupport(drivername string) error {
	if strings.HasPrefix(drivername, lxc.DriverName) {
		return lxc.ErrExec
	}
	return nil
}
