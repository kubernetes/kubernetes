// Package devices contains functionality to manage cgroup devices, which
// is exposed indirectly via libcontainer/cgroups managers.
//
// To enable cgroup managers to manage devices, this package must be imported.
package devices

import (
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/systemd"
)

func init() {
	cgroups.DevicesSetV1 = setV1
	cgroups.DevicesSetV2 = setV2
	systemd.GenerateDeviceProps = systemdProperties
}
