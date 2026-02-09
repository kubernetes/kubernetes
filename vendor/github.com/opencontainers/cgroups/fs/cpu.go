package fs

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strconv"

	"github.com/opencontainers/cgroups"
	"github.com/opencontainers/cgroups/fscommon"
	"golang.org/x/sys/unix"
)

type CpuGroup struct{}

func (s *CpuGroup) Name() string {
	return "cpu"
}

func (s *CpuGroup) Apply(path string, r *cgroups.Resources, pid int) error {
	if err := os.MkdirAll(path, 0o755); err != nil {
		return err
	}
	// We should set the real-Time group scheduling settings before moving
	// in the process because if the process is already in SCHED_RR mode
	// and no RT bandwidth is set, adding it will fail.
	if err := s.SetRtSched(path, r); err != nil {
		return err
	}
	// Since we are not using apply(), we need to place the pid
	// into the procs file.
	return cgroups.WriteCgroupProc(path, pid)
}

func (s *CpuGroup) SetRtSched(path string, r *cgroups.Resources) error {
	var period string
	if r.CpuRtPeriod != 0 {
		period = strconv.FormatUint(r.CpuRtPeriod, 10)
		if err := cgroups.WriteFile(path, "cpu.rt_period_us", period); err != nil {
			// The values of cpu.rt_period_us and cpu.rt_runtime_us
			// are inter-dependent and need to be set in a proper order.
			// If the kernel rejects the new period value with EINVAL
			// and the new runtime value is also being set, let's
			// ignore the error for now and retry later.
			if !errors.Is(err, unix.EINVAL) || r.CpuRtRuntime == 0 {
				return err
			}
		} else {
			period = ""
		}
	}
	if r.CpuRtRuntime != 0 {
		if err := cgroups.WriteFile(path, "cpu.rt_runtime_us", strconv.FormatInt(r.CpuRtRuntime, 10)); err != nil {
			return err
		}
		if period != "" {
			if err := cgroups.WriteFile(path, "cpu.rt_period_us", period); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *CpuGroup) Set(path string, r *cgroups.Resources) error {
	if r.CpuShares != 0 {
		shares := r.CpuShares
		if err := cgroups.WriteFile(path, "cpu.shares", strconv.FormatUint(shares, 10)); err != nil {
			return err
		}
		// read it back
		sharesRead, err := fscommon.GetCgroupParamUint(path, "cpu.shares")
		if err != nil {
			return err
		}
		// ... and check
		if shares > sharesRead {
			return fmt.Errorf("the maximum allowed cpu-shares is %d", sharesRead)
		} else if shares < sharesRead {
			return fmt.Errorf("the minimum allowed cpu-shares is %d", sharesRead)
		}
	}

	var period string
	if r.CpuPeriod != 0 {
		period = strconv.FormatUint(r.CpuPeriod, 10)
		if err := cgroups.WriteFile(path, "cpu.cfs_period_us", period); err != nil {
			// Sometimes when the period to be set is smaller
			// than the current one, it is rejected by the kernel
			// (EINVAL) as old_quota/new_period exceeds the parent
			// cgroup quota limit. If this happens and the quota is
			// going to be set, ignore the error for now and retry
			// after setting the quota.
			if !errors.Is(err, unix.EINVAL) || r.CpuQuota == 0 {
				return err
			}
		} else {
			period = ""
		}
	}

	var burst string
	if r.CpuBurst != nil {
		burst = strconv.FormatUint(*r.CpuBurst, 10)
		if err := cgroups.WriteFile(path, "cpu.cfs_burst_us", burst); err != nil {
			if errors.Is(err, unix.ENOENT) {
				// If CPU burst knob is not available (e.g.
				// older kernel), ignore it.
				burst = ""
			} else {
				// Sometimes when the burst to be set is larger
				// than the current one, it is rejected by the kernel
				// (EINVAL) as old_quota/new_burst exceeds the parent
				// cgroup quota limit. If this happens and the quota is
				// going to be set, ignore the error for now and retry
				// after setting the quota.
				if !errors.Is(err, unix.EINVAL) || r.CpuQuota == 0 {
					return err
				}
			}
		} else {
			burst = ""
		}
	}
	if r.CpuQuota != 0 {
		if err := cgroups.WriteFile(path, "cpu.cfs_quota_us", strconv.FormatInt(r.CpuQuota, 10)); err != nil {
			return err
		}
		if period != "" {
			if err := cgroups.WriteFile(path, "cpu.cfs_period_us", period); err != nil {
				return err
			}
		}
		if burst != "" {
			if err := cgroups.WriteFile(path, "cpu.cfs_burst_us", burst); err != nil {
				return err
			}
		}
	}

	if r.CPUIdle != nil {
		idle := strconv.FormatInt(*r.CPUIdle, 10)
		if err := cgroups.WriteFile(path, "cpu.idle", idle); err != nil {
			return err
		}
	}

	return s.SetRtSched(path, r)
}

func (s *CpuGroup) GetStats(path string, stats *cgroups.Stats) error {
	const file = "cpu.stat"
	f, err := cgroups.OpenFile(path, file, os.O_RDONLY)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		t, v, err := fscommon.ParseKeyValue(sc.Text())
		if err != nil {
			return &parseError{Path: path, File: file, Err: err}
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
