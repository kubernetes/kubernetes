package fs

import (
	"github.com/opencontainers/cgroups"
)

type PerfEventGroup struct{}

func (s *PerfEventGroup) Name() string {
	return "perf_event"
}

// ID returns the controller ID for perf_event subsystem.
// Returns 0 as perf_event is not a cgroups.Controller.
func (s *PerfEventGroup) ID() cgroups.Controller {
	return 0
}

func (s *PerfEventGroup) Apply(path string, _ *cgroups.Resources, pid int) error {
	return apply(path, pid)
}

func (s *PerfEventGroup) Set(_ string, _ *cgroups.Resources) error {
	return nil
}

func (s *PerfEventGroup) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}
