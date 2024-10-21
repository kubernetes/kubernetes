package fs

import (
	"github.com/dims/libcontainer/cgroups"
	"github.com/dims/libcontainer/cgroups/fscommon"
	"github.com/dims/libcontainer/configs"
)

type RdmaGroup struct{}

func (s *RdmaGroup) Name() string {
	return "rdma"
}

func (s *RdmaGroup) Apply(path string, _ *configs.Resources, pid int) error {
	return apply(path, pid)
}

func (s *RdmaGroup) Set(path string, r *configs.Resources) error {
	return fscommon.RdmaSet(path, r)
}

func (s *RdmaGroup) GetStats(path string, stats *cgroups.Stats) error {
	return fscommon.RdmaGetStats(path, stats)
}
