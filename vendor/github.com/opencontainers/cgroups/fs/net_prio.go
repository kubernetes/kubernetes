package fs

import (
	"github.com/opencontainers/cgroups"
)

type NetPrioGroup struct{}

func (s *NetPrioGroup) Name() string {
	return "net_prio"
}

// ID returns the controller ID for net_prio subsystem.
// Returns 0 as net_prio is not a cgroups.Controller.
func (s *NetPrioGroup) ID() cgroups.Controller {
	return 0
}

func (s *NetPrioGroup) Apply(path string, _ *cgroups.Resources, pid int) error {
	return apply(path, pid)
}

func (s *NetPrioGroup) Set(path string, r *cgroups.Resources) error {
	for _, prioMap := range r.NetPrioIfpriomap {
		if err := cgroups.WriteFile(path, "net_prio.ifpriomap", prioMap.CgroupString()); err != nil {
			return err
		}
	}

	return nil
}

func (s *NetPrioGroup) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}
