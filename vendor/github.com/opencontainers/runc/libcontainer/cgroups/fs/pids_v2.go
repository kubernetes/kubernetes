// +build linux

package fs

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
	"golang.org/x/sys/unix"
)

type PidsGroupV2 struct {
}

func (s *PidsGroupV2) Name() string {
	return "pids"
}

func (s *PidsGroupV2) Apply(d *cgroupData) error {
	_, err := d.join("pids")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}

func (s *PidsGroupV2) Set(path string, cgroup *configs.Cgroup) error {
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

func (s *PidsGroupV2) Remove(d *cgroupData) error {
	return removePath(d.path("pids"))
}

func isNOTSUP(err error) bool {
	switch err := err.(type) {
	case *os.PathError:
		return err.Err == unix.ENOTSUP
	default:
		return false
	}
}

func (s *PidsGroupV2) GetStats(path string, stats *cgroups.Stats) error {
	current, err := getCgroupParamUint(path, "pids.current")
	if os.IsNotExist(err) {
		// if the controller is not enabled, let's read the list
		// PIDs (or threads if cgroup.threads is enabled)
		contents, err := ioutil.ReadFile(filepath.Join(path, "cgroup.procs"))
		if err != nil && isNOTSUP(err) {
			contents, err = ioutil.ReadFile(filepath.Join(path, "cgroup.threads"))
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
	if err != nil {
		return fmt.Errorf("failed to parse pids.current - %s", err)
	}

	maxString, err := getCgroupParamString(path, "pids.max")
	if err != nil {
		return fmt.Errorf("failed to parse pids.max - %s", err)
	}

	// Default if pids.max == "max" is 0 -- which represents "no limit".
	var max uint64
	if maxString != "max" {
		max, err = parseUint(maxString, 10, 64)
		if err != nil {
			return fmt.Errorf("failed to parse pids.max - unable to parse %q as a uint from Cgroup file %q", maxString, filepath.Join(path, "pids.max"))
		}
	}

	stats.PidsStats.Current = current
	stats.PidsStats.Limit = max
	return nil
}
