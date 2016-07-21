// +build linux

package fs

import (
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type NetClsGroup struct {
}

func (s *NetClsGroup) Name() string {
	return "net_cls"
}

func (s *NetClsGroup) Apply(d *cgroupData) error {
	_, err := d.join("net_cls")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}

func (s *NetClsGroup) Set(path string, cgroup *configs.Cgroup) error {
	if cgroup.Resources.NetClsClassid != "" {
		if err := writeFile(path, "net_cls.classid", cgroup.Resources.NetClsClassid); err != nil {
			return err
		}
	}

	return nil
}

func (s *NetClsGroup) Remove(d *cgroupData) error {
	return removePath(d.path("net_cls"))
}

func (s *NetClsGroup) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}
