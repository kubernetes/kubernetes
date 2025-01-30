package fs2

import (
	"bufio"
	"errors"
	"os"
	"strconv"

	"golang.org/x/sys/unix"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

func isCpuSet(r *configs.Resources) bool {
	return r.CpuWeight != 0 || r.CpuQuota != 0 || r.CpuPeriod != 0 || r.CPUIdle != nil || r.CpuBurst != nil
}

func setCpu(dirPath string, r *configs.Resources) error {
	if !isCpuSet(r) {
		return nil
	}

	if r.CPUIdle != nil {
		if err := cgroups.WriteFile(dirPath, "cpu.idle", strconv.FormatInt(*r.CPUIdle, 10)); err != nil {
			return err
		}
	}

	// NOTE: .CpuShares is not used here. Conversion is the caller's responsibility.
	if r.CpuWeight != 0 {
		if err := cgroups.WriteFile(dirPath, "cpu.weight", strconv.FormatUint(r.CpuWeight, 10)); err != nil {
			return err
		}
	}

	var burst string
	if r.CpuBurst != nil {
		burst = strconv.FormatUint(*r.CpuBurst, 10)
		if err := cgroups.WriteFile(dirPath, "cpu.max.burst", burst); err != nil {
			// Sometimes when the burst to be set is larger
			// than the current one, it is rejected by the kernel
			// (EINVAL) as old_quota/new_burst exceeds the parent
			// cgroup quota limit. If this happens and the quota is
			// going to be set, ignore the error for now and retry
			// after setting the quota.
			if !errors.Is(err, unix.EINVAL) || r.CpuQuota == 0 {
				return err
			}
		} else {
			burst = ""
		}
	}
	if r.CpuQuota != 0 || r.CpuPeriod != 0 {
		str := "max"
		if r.CpuQuota > 0 {
			str = strconv.FormatInt(r.CpuQuota, 10)
		}
		period := r.CpuPeriod
		if period == 0 {
			// This default value is documented in
			// https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
			period = 100000
		}
		str += " " + strconv.FormatUint(period, 10)
		if err := cgroups.WriteFile(dirPath, "cpu.max", str); err != nil {
			return err
		}
		if burst != "" {
			if err := cgroups.WriteFile(dirPath, "cpu.max.burst", burst); err != nil {
				return err
			}
		}
	}

	return nil
}

func statCpu(dirPath string, stats *cgroups.Stats) error {
	const file = "cpu.stat"
	f, err := cgroups.OpenFile(dirPath, file, os.O_RDONLY)
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		t, v, err := fscommon.ParseKeyValue(sc.Text())
		if err != nil {
			return &parseError{Path: dirPath, File: file, Err: err}
		}
		switch t {
		case "usage_usec":
			stats.CpuStats.CpuUsage.TotalUsage = v * 1000

		case "user_usec":
			stats.CpuStats.CpuUsage.UsageInUsermode = v * 1000

		case "system_usec":
			stats.CpuStats.CpuUsage.UsageInKernelmode = v * 1000

		case "nr_periods":
			stats.CpuStats.ThrottlingData.Periods = v

		case "nr_throttled":
			stats.CpuStats.ThrottlingData.ThrottledPeriods = v

		case "throttled_usec":
			stats.CpuStats.ThrottlingData.ThrottledTime = v * 1000
		}
	}
	if err := sc.Err(); err != nil {
		return &parseError{Path: dirPath, File: file, Err: err}
	}
	return nil
}
