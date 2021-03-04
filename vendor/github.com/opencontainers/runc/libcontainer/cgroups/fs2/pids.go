// +build linux

package fs2

import (
	"io/ioutil"
	"path/filepath"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

func isPidsSet(cgroup *configs.Cgroup) bool {
	return cgroup.Resources.PidsLimit != 0
}

func setPids(dirPath string, cgroup *configs.Cgroup) error {
	if !isPidsSet(cgroup) {
		return nil
	}
	if val := numToStr(cgroup.Resources.PidsLimit); val != "" {
		if err := fscommon.WriteFile(dirPath, "pids.max", val); err != nil {
			return err
		}
	}

	return nil
}

func statPidsWithoutController(dirPath string, stats *cgroups.Stats) error {
	// if the controller is not enabled, let's read PIDS from cgroups.procs
	// (or threads if cgroup.threads is enabled)
	contents, err := ioutil.ReadFile(filepath.Join(dirPath, "cgroup.procs"))
	if errors.Is(err, unix.ENOTSUP) {
		contents, err = ioutil.ReadFile(filepath.Join(dirPath, "cgroup.threads"))
	}
	if err != nil {
		return err
	}
	pids := make(map[string]string)
	for _, i := range strings.Split(string(contents), "\n") {
		if i != "" {
			pids[i] = i
		}
	}
	stats.PidsStats.Current = uint64(len(pids))
	stats.PidsStats.Limit = 0
	return nil
}

func statPids(dirPath string, stats *cgroups.Stats) error {
	current, err := fscommon.GetCgroupParamUint(dirPath, "pids.current")
	if err != nil {
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
