// +build !linux

package systemd

import (
	"fmt"

	"github.com/docker/libcontainer/cgroups"
)

func UseSystemd() bool {
	return false
}

func Apply(c *cgroups.Cgroup, pid int) (map[string]string, error) {
	return nil, fmt.Errorf("Systemd not supported")
}

func GetPids(c *cgroups.Cgroup) ([]int, error) {
	return nil, fmt.Errorf("Systemd not supported")
}

func ApplyDevices(c *cgroups.Cgroup, pid int) error {
	return fmt.Errorf("Systemd not supported")
}

func Freeze(c *cgroups.Cgroup, state cgroups.FreezerState) error {
	return fmt.Errorf("Systemd not supported")
}
