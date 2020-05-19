// +build linux

package fs2

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

func setPids(dirPath string, cgroup *configs.Cgroup) error {
	if cgroup.Resources.PidsLimit != 0 {
		// "max" is the fallback value.
		limit := "max"

		if cgroup.Resources.PidsLimit > 0 {
			limit = strconv.FormatInt(cgroup.Resources.PidsLimit, 10)
		}

		if err := fscommon.WriteFile(dirPath, "pids.max", limit); err != nil {
			return err
		}
	}

	return nil
}

func isNOTSUP(err error) bool {
	switch err := err.(type) {
	case *os.PathError:
		return err.Err == unix.ENOTSUP
	default:
		return false
	}
}

func statPidsWithoutController(dirPath string, stats *cgroups.Stats) error {
	// if the controller is not enabled, let's read PIDS from cgroups.procs
	// (or threads if cgroup.threads is enabled)
	contents, err := ioutil.ReadFile(filepath.Join(dirPath, "cgroup.procs"))
	if err != nil && isNOTSUP(err) {
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
