package fs

import (
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type DevicesGroup struct{}

func (s *DevicesGroup) Name() string {
	return "devices"
}

func (s *DevicesGroup) Apply(path string, r *configs.Resources, pid int) error {
	if r.SkipDevices {
		return nil
	}
	if path == "" {
		// Return error here, since devices cgroup
		// is a hard requirement for container's security.
		return errSubsystemDoesNotExist
	}

	return apply(path, pid)
}

func (s *DevicesGroup) Set(path string, r *configs.Resources) error {
	if cgroups.DevicesSetV1 == nil {
		if len(r.Devices) == 0 {
			return nil
		}
		return cgroups.ErrDevicesUnsupported
	}
	return cgroups.DevicesSetV1(path, r)
}

func (s *DevicesGroup) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}
