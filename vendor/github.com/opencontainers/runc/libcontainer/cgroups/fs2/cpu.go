// +build linux

package fs2

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

func setCpu(dirPath string, cgroup *configs.Cgroup) error {
	if cgroup.Resources.CpuWeight != 0 {
		if err := fscommon.WriteFile(dirPath, "cpu.weight", strconv.FormatUint(cgroup.Resources.CpuWeight, 10)); err != nil {
			return err
		}
	}

	if cgroup.Resources.CpuMax != "" {
		if err := fscommon.WriteFile(dirPath, "cpu.max", cgroup.Resources.CpuMax); err != nil {
			return err
		}
	}

	return nil
}
func statCpu(dirPath string, stats *cgroups.Stats) error {
	f, err := os.Open(filepath.Join(dirPath, "cpu.stat"))
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		t, v, err := fscommon.GetCgroupParamKeyValue(sc.Text())
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
