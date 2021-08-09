// +build linux

package fs

import (
	"fmt"
	"path/filepath"
	"strconv"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type PidsGroup struct{}

func (s *PidsGroup) Name() string {
	return "pids"
}

func (s *PidsGroup) Apply(path string, d *cgroupData) error {
	return join(path, d.pid)
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
		return fmt.Errorf("failed to parse pids.current - %s", err)
	}

	maxString, err := fscommon.GetCgroupParamString(path, "pids.max")
	if err != nil {
		return fmt.Errorf("failed to parse pids.max - %s", err)
	}

	// Default if pids.max == "max" is 0 -- which represents "no limit".
	var max uint64
	if maxString != "max" {
		max, err = fscommon.ParseUint(maxString, 10, 64)
		if err != nil {
			return fmt.Errorf("failed to parse pids.max - unable to parse %q as a uint from Cgroup file %q", maxString, filepath.Join(path, "pids.max"))
		}
	}

	stats.PidsStats.Current = current
	stats.PidsStats.Limit = max
	return nil
}
