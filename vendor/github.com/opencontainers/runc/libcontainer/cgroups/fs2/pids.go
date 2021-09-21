// +build linux

package fs2

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

func isPidsSet(r *configs.Resources) bool {
	return r.PidsLimit != 0
}

func setPids(dirPath string, r *configs.Resources) error {
	if !isPidsSet(r) {
		return nil
	}
	if val := numToStr(r.PidsLimit); val != "" {
		if err := cgroups.WriteFile(dirPath, "pids.max", val); err != nil {
			return err
		}
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
		return errors.Wrap(err, "failed to parse pids.current")
	}

	maxString, err := fscommon.GetCgroupParamString(dirPath, "pids.max")
	if err != nil {
		return errors.Wrap(err, "failed to parse pids.max")
	}

	// Default if pids.max == "max" is 0 -- which represents "no limit".
	var max uint64
	if maxString != "max" {
		max, err = fscommon.ParseUint(maxString, 10, 64)
		if err != nil {
			return errors.Wrapf(err, "failed to parse pids.max - unable to parse %q as a uint from Cgroup file %q",
				maxString, filepath.Join(dirPath, "pids.max"))
		}
	}

	stats.PidsStats.Current = current
	stats.PidsStats.Limit = max
	return nil
}
