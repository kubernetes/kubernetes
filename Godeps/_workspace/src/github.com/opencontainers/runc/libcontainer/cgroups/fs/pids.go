// +build linux

package fs

import (
	"fmt"
	"strconv"

	"github.com/opencontainers/runc/libcontainer/cgroups"
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

		if err := writeFile(path, "pids.max", limit); err != nil {
			return err
		}
	}

	return nil
}

func (s *PidsGroup) Remove(d *cgroupData) error {
	return removePath(d.path("pids"))
}

func (s *PidsGroup) GetStats(path string, stats *cgroups.Stats) error {
	value, err := getCgroupParamUint(path, "pids.current")
	if err != nil {
		return fmt.Errorf("failed to parse pids.current - %s", err)
	}

	stats.PidsStats.Current = value
	return nil
}
