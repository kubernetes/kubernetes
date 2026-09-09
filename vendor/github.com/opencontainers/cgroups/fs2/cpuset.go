package fs2

import (
	"github.com/opencontainers/cgroups"
)

func isCpusetSet(r *cgroups.Resources) bool {
	return r.CpusetCpus != "" || r.CpusetMems != ""
}

func setCpuset(dirPath string, r *cgroups.Resources) error {
	if !isCpusetSet(r) {
		return nil
	}

	if r.CpusetCpus != "" {
		if err := cgroups.WriteFile(dirPath, "cpuset.cpus", r.CpusetCpus); err != nil {
			return err
		}
	}
	if r.CpusetMems != "" {
		if err := cgroups.WriteFile(dirPath, "cpuset.mems", r.CpusetMems); err != nil {
			return err
		}
	}
	return nil
}
