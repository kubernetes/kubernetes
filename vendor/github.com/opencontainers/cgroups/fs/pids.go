package fs

import (
	"math"
	"strconv"

	"github.com/opencontainers/cgroups"
	"github.com/opencontainers/cgroups/fscommon"
)

type PidsGroup struct{}

func (s *PidsGroup) Name() string {
	return "pids"
}

func (s *PidsGroup) Apply(path string, _ *cgroups.Resources, pid int) error {
	return apply(path, pid)
}

func (s *PidsGroup) Set(path string, r *cgroups.Resources) error {
	if r.PidsLimit == nil {
		return nil
	}

	// "max" is the fallback value.
	val := "max"
	if limit := *r.PidsLimit; limit > 0 {
		val = strconv.FormatInt(limit, 10)
	} else if limit == 0 {
		// systemd doesn't support setting pids.max to "0", so when setting
		// TasksMax we need to remap it to "1". We do the same thing here to
		// avoid flip-flop behaviour between the fs and systemd drivers. In
		// practice, the pids cgroup behaviour is basically identical.
		val = "1"
	}
	if err := cgroups.WriteFile(path, "pids.max", val); err != nil {
		return err
	}
	return nil
}

func (s *PidsGroup) GetStats(path string, stats *cgroups.Stats) error {
	if !cgroups.PathExists(path) {
		return nil
	}
	current, err := fscommon.GetCgroupParamUint(path, "pids.current")
	if err != nil {
		return err
	}

	max, err := fscommon.GetCgroupParamUint(path, "pids.max")
	if err != nil {
		return err
	}
	// If no limit is set, read from pids.max returns "max", which is
	// converted to MaxUint64 by GetCgroupParamUint. Historically, we
	// represent "no limit" for pids as 0, thus this conversion.
	if max == math.MaxUint64 {
		max = 0
	}

	stats.PidsStats.Current = current
	stats.PidsStats.Limit = max
	return nil
}
