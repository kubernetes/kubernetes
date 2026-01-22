package fs

import (
	"github.com/opencontainers/cgroups"
)

type NameGroup struct {
	GroupName string
	Join      bool
}

func (s *NameGroup) Name() string {
	return s.GroupName
}

func (s *NameGroup) Apply(path string, _ *cgroups.Resources, pid int) error {
	if s.Join {
		// Ignore errors if the named cgroup does not exist.
		_ = apply(path, pid)
	}
	return nil
}

func (s *NameGroup) Set(_ string, _ *cgroups.Resources) error {
	return nil
}

func (s *NameGroup) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}
