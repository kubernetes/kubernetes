package fs

import (
	"github.com/opencontainers/cgroups"
)

type DevicesGroup struct{}

func (s *DevicesGroup) Name() string {
	return "devices"
}

// ID returns the controller ID for devices subsystem.
// Returns 0 as devices is not a cgroups.Controller.
func (s *DevicesGroup) ID() cgroups.Controller {
	return 0
}

func (s *DevicesGroup) Apply(path string, r *cgroups.Resources, pid int) error {
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

func (s *DevicesGroup) Set(path string, r *cgroups.Resources) error {
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
