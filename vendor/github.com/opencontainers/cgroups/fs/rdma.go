package fs

import (
	"github.com/opencontainers/cgroups"
	"github.com/opencontainers/cgroups/fscommon"
)

type RdmaGroup struct{}

func (s *RdmaGroup) Name() string {
	return "rdma"
}

// ID returns the controller ID for rdma subsystem.
func (s *RdmaGroup) ID() cgroups.Controller {
	return cgroups.RDMA
}

func (s *RdmaGroup) Apply(path string, _ *cgroups.Resources, pid int) error {
	return apply(path, pid)
}

func (s *RdmaGroup) Set(path string, r *cgroups.Resources) error {
	return fscommon.RdmaSet(path, r)
}

func (s *RdmaGroup) GetStats(path string, stats *cgroups.Stats) error {
	return fscommon.RdmaGetStats(path, stats)
}
