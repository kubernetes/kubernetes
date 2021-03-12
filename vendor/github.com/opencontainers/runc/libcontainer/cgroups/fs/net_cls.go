// +build linux

package fs

import (
	"strconv"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type NetClsGroup struct {
}

func (s *NetClsGroup) Name() string {
	return "net_cls"
}

func (s *NetClsGroup) Apply(path string, d *cgroupData) error {
	return join(path, d.pid)
}

func (s *NetClsGroup) Set(path string, cgroup *configs.Cgroup) error {
	if cgroup.Resources.NetClsClassid != 0 {
		if err := fscommon.WriteFile(path, "net_cls.classid", strconv.FormatUint(uint64(cgroup.Resources.NetClsClassid), 10)); err != nil {
			return err
		}
	}

	return nil
}

func (s *NetClsGroup) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}
