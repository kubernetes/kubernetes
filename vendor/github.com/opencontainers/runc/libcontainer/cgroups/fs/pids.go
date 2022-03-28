package fs

import (
	"math"
	"strconv"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type PidsGroup struct{}

func (s *PidsGroup) Name() string {
	return "pids"
}

func (s *PidsGroup) Apply(path string, _ *configs.Resources, pid int) error {
	return apply(path, pid)
}

func (s *PidsGroup) Set(path string, r *configs.Resources) error {
	if r.PidsLimit != 0 {
		// "max" is the fallback value.
		limit := "max"

		if r.PidsLimit > 0 {
			limit = strconv.FormatInt(r.PidsLimit, 10)
		}

		if err := cgroups.WriteFile(path, "pids.max", limit); err != nil {
			return err
		}
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
