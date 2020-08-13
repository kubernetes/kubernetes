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

type PidsGroup struct {
}

func (s *PidsGroup) Name() string {
	return "pids"
}

func (s *PidsGroup) Apply(d *cgroupData) error {
	_, err := d.join("pids")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}

func (s *PidsGroup) Set(path string, cgroup *configs.Cgroup) error {
	if cgroup.Resources.PidsLimit != 0 {
		// "max" is the fallback value.
		limit := "max"

		if cgroup.Resources.PidsLimit > 0 {
			limit = strconv.FormatInt(cgroup.Resources.PidsLimit, 10)
		}

		if err := fscommon.WriteFile(path, "pids.max", limit); err != nil {
			return err
		}
	}

	return nil
}

func (s *PidsGroup) Remove(d *cgroupData) error {
	return removePath(d.path("pids"))
}

func (s *PidsGroup) GetStats(path string, stats *cgroups.Stats) error {
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
