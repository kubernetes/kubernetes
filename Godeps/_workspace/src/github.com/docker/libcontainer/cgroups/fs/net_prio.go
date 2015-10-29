package fs

import (
	"github.com/docker/libcontainer/cgroups"
	"github.com/docker/libcontainer/configs"
)

type NetPrioGroup struct {
}

func (s *NetPrioGroup) Apply(d *data) error {
	dir, err := d.join("net_prio")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}

	if err := s.Set(dir, d.c); err != nil {
		return err
	}

	return nil
}

func (s *NetPrioGroup) Set(path string, cgroup *configs.Cgroup) error {
	for _, prioMap := range cgroup.NetPrioIfpriomap {
		if err := writeFile(path, "net_prio.ifpriomap", prioMap.CgroupString()); err != nil {
			return err
		}
	}

	return nil
}

func (s *NetPrioGroup) Remove(d *data) error {
	return removePath(d.path("net_prio"))
}

func (s *NetPrioGroup) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}
