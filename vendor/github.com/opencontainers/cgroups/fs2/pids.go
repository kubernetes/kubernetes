package fs2

import (
	"errors"
	"math"
	"os"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"

	"github.com/opencontainers/cgroups"
	"github.com/opencontainers/cgroups/fscommon"
)

func isPidsSet(r *cgroups.Resources) bool {
	return r.PidsLimit != nil
}

func setPids(dirPath string, r *cgroups.Resources) error {
	if !isPidsSet(r) {
		return nil
	}
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
	if err := cgroups.WriteFile(dirPath, "pids.max", val); err != nil {
		return err
	}
	return nil
}

func statPidsFromCgroupProcs(dirPath string, stats *cgroups.Stats) error {
	// if the controller is not enabled, let's read PIDS from cgroups.procs
	// (or threads if cgroup.threads is enabled)
	contents, err := cgroups.ReadFile(dirPath, "cgroup.procs")
	if errors.Is(err, unix.ENOTSUP) {
		contents, err = cgroups.ReadFile(dirPath, "cgroup.threads")
	}
	if err != nil {
		return err
	}
	pids := strings.Count(contents, "\n")
	stats.PidsStats.Current = uint64(pids)
	stats.PidsStats.Limit = 0
	return nil
}

func statPids(dirPath string, stats *cgroups.Stats) error {
	current, err := fscommon.GetCgroupParamUint(dirPath, "pids.current")
	if err != nil {
		if os.IsNotExist(err) {
			return statPidsFromCgroupProcs(dirPath, stats)
		}
		return err
	}

	max, err := fscommon.GetCgroupParamUint(dirPath, "pids.max")
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
