// +build linux

package fs

import (
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type NameGroup struct {
	GroupName string
	Join      bool
}

func (s *NameGroup) Name() string {
	return s.GroupName
}

func (s *NameGroup) Apply(path string, d *cgroupData) error {
	if s.Join {
		// ignore errors if the named cgroup does not exist
		_ = join(path, d.pid)
	}
	return nil
}

func (s *NameGroup) Set(_ string, _ *configs.Resources) error {
	return nil
}

func (s *NameGroup) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}
