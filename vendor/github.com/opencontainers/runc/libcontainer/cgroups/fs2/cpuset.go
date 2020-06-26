// +build linux

package fs2

import (
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

func isCpusetSet(cgroup *configs.Cgroup) bool {
	return cgroup.Resources.CpusetCpus != "" || cgroup.Resources.CpusetMems != ""
}

func setCpuset(dirPath string, cgroup *configs.Cgroup) error {
	if !isCpusetSet(cgroup) {
		return nil
	}

	if cgroup.Resources.CpusetCpus != "" {
		if err := fscommon.WriteFile(dirPath, "cpuset.cpus", cgroup.Resources.CpusetCpus); err != nil {
			return err
		}
	}
	if cgroup.Resources.CpusetMems != "" {
		if err := fscommon.WriteFile(dirPath, "cpuset.mems", cgroup.Resources.CpusetMems); err != nil {
			return err
		}
	}
	return nil
}
