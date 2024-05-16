package fs

import (
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type NetPrioGroup struct{}

func (s *NetPrioGroup) Name() string {
	return "net_prio"
}

func (s *NetPrioGroup) Apply(path string, _ *configs.Resources, pid int) error {
	return apply(path, pid)
}

func (s *NetPrioGroup) Set(path string, r *configs.Resources) error {
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
