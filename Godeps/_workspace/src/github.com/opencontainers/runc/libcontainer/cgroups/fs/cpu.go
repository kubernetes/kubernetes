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

type CpuGroup struct {
}

func (s *CpuGroup) Name() string {
	return "cpu"
}

func (s *CpuGroup) Apply(d *cgroupData) error {
	// We always want to join the cpu group, to allow fair cpu scheduling
	// on a container basis
	dir, err := d.join("cpu")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}

	if err := s.Set(dir, d.config); err != nil {
		return err
	}

	return nil
}

func (s *CpuGroup) Set(path string, cgroup *configs.Cgroup) error {
	if cgroup.CpuShares != 0 {
		if err := writeFile(path, "cpu.shares", strconv.FormatInt(cgroup.CpuShares, 10)); err != nil {
			return err
		}
	}
	if cgroup.CpuPeriod != 0 {
		if err := writeFile(path, "cpu.cfs_period_us", strconv.FormatInt(cgroup.CpuPeriod, 10)); err != nil {
			return err
		}
	}
	if cgroup.CpuQuota != 0 {
		if err := writeFile(path, "cpu.cfs_quota_us", strconv.FormatInt(cgroup.CpuQuota, 10)); err != nil {
			return err
		}
	}
	if cgroup.CpuRtPeriod != 0 {
		if err := writeFile(path, "cpu.rt_period_us", strconv.FormatInt(cgroup.CpuRtPeriod, 10)); err != nil {
			return err
		}
	}
	if cgroup.CpuRtRuntime != 0 {
		if err := writeFile(path, "cpu.rt_runtime_us", strconv.FormatInt(cgroup.CpuRtRuntime, 10)); err != nil {
			return err
		}
	}

	return nil
}

func (s *CpuGroup) Remove(d *cgroupData) error {
	return removePath(d.path("cpu"))
}

func (s *CpuGroup) GetStats(path string, stats *cgroups.Stats) error {
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
		case "nr_periods":
			stats.CpuStats.ThrottlingData.Periods = v

		case "nr_throttled":
			stats.CpuStats.ThrottlingData.ThrottledPeriods = v

		case "throttled_time":
			stats.CpuStats.ThrottlingData.ThrottledTime = v
		}
	}
	return nil
}
