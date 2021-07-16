// +build linux

package fs

import (
	"fmt"
	"strconv"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type HugetlbGroup struct{}

func (s *HugetlbGroup) Name() string {
	return "hugetlb"
}

func (s *HugetlbGroup) Apply(path string, d *cgroupData) error {
	return join(path, d.pid)
}

func (s *HugetlbGroup) Set(path string, r *configs.Resources) error {
	for _, hugetlb := range r.HugetlbLimit {
		if err := cgroups.WriteFile(path, "hugetlb."+hugetlb.Pagesize+".limit_in_bytes", strconv.FormatUint(hugetlb.Limit, 10)); err != nil {
			return err
		}
	}

	return nil
}

func (s *HugetlbGroup) GetStats(path string, stats *cgroups.Stats) error {
	hugetlbStats := cgroups.HugetlbStats{}
	if !cgroups.PathExists(path) {
		return nil
	}
	for _, pageSize := range HugePageSizes {
		usage := "hugetlb." + pageSize + ".usage_in_bytes"
		value, err := fscommon.GetCgroupParamUint(path, usage)
		if err != nil {
			return fmt.Errorf("failed to parse %s - %v", usage, err)
		}
		hugetlbStats.Usage = value

		maxUsage := "hugetlb." + pageSize + ".max_usage_in_bytes"
		value, err = fscommon.GetCgroupParamUint(path, maxUsage)
		if err != nil {
			return fmt.Errorf("failed to parse %s - %v", maxUsage, err)
		}
		hugetlbStats.MaxUsage = value

		failcnt := "hugetlb." + pageSize + ".failcnt"
		value, err = fscommon.GetCgroupParamUint(path, failcnt)
		if err != nil {
			return fmt.Errorf("failed to parse %s - %v", failcnt, err)
		}
		hugetlbStats.Failcnt = value

		stats.HugetlbStats[pageSize] = hugetlbStats
	}

	return nil
}
