// +build linux

package fs

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type CpuGroupV2 struct {
}

func (s *CpuGroupV2) Name() string {
	return "cpu"
}

func (s *CpuGroupV2) Apply(d *cgroupData) error {
	// We always want to join the cpu group, to allow fair cpu scheduling
	// on a container basis
	path, err := d.path("cpu")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return s.ApplyDir(path, d.config, d.pid)
}

func (s *CpuGroupV2) ApplyDir(path string, cgroup *configs.Cgroup, pid int) error {
	// This might happen if we have no cpu cgroup mounted.
	// Just do nothing and don't fail.
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(path, 0755); err != nil {
		return err
	}
	return cgroups.WriteCgroupProc(path, pid)
}

func (s *CpuGroupV2) Set(path string, cgroup *configs.Cgroup) error {
	if cgroup.Resources.CpuWeight != 0 {
		if err := writeFile(path, "cpu.weight", strconv.FormatUint(cgroup.Resources.CpuWeight, 10)); err != nil {
			return err
		}
	}

	if cgroup.Resources.CpuMax != "" {
		if err := writeFile(path, "cpu.max", cgroup.Resources.CpuMax); err != nil {
			return err
		}
	}

	return nil
}

func (s *CpuGroupV2) Remove(d *cgroupData) error {
	return removePath(d.path("cpu"))
}

func (s *CpuGroupV2) GetStats(path string, stats *cgroups.Stats) error {
	f, err := os.Open(filepath.Join(path, "cpu.stat"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		t, v, err := getCgroupParamKeyValue(sc.Text())
		if err != nil {
			return err
		}
		switch t {
		case "usage_usec":
			stats.CpuStats.CpuUsage.TotalUsage = v * 1000

		case "user_usec":
			stats.CpuStats.CpuUsage.UsageInUsermode = v * 1000

		case "system_usec":
			stats.CpuStats.CpuUsage.UsageInKernelmode = v * 1000
		}
	}
	return nil
}
