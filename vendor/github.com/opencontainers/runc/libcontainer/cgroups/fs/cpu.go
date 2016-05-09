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
	_, err := d.join("cpu")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}

func (s *CpuGroup) Set(path string, cgroup *configs.Cgroup) error {
	if cgroup.Resources.CpuShares != 0 {
		if err := writeFile(path, "cpu.shares", strconv.FormatInt(cgroup.Resources.CpuShares, 10)); err != nil {
			return err
		}
	}
	if cgroup.Resources.CpuPeriod != 0 {
		if err := writeFile(path, "cpu.cfs_period_us", strconv.FormatInt(cgroup.Resources.CpuPeriod, 10)); err != nil {
			return err
		}
	}
	if cgroup.Resources.CpuQuota != 0 {
		if err := writeFile(path, "cpu.cfs_quota_us", strconv.FormatInt(cgroup.Resources.CpuQuota, 10)); err != nil {
			return err
		}
	}
	if cgroup.Resources.CpuRtPeriod != 0 {
		if err := writeFile(path, "cpu.rt_period_us", strconv.FormatInt(cgroup.Resources.CpuRtPeriod, 10)); err != nil {
			return err
		}
	}
	if cgroup.Resources.CpuRtRuntime != 0 {
		if err := writeFile(path, "cpu.rt_runtime_us", strconv.FormatInt(cgroup.Resources.CpuRtRuntime, 10)); err != nil {
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
