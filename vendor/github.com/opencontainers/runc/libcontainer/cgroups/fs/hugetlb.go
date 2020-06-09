// +build linux

package fs

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type HugetlbGroup struct {
}

func (s *HugetlbGroup) Name() string {
	return "hugetlb"
}

func (s *HugetlbGroup) Apply(d *cgroupData) error {
	_, err := d.join("hugetlb")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}

func (s *HugetlbGroup) Set(path string, cgroup *configs.Cgroup) error {
	for _, hugetlb := range cgroup.Resources.HugetlbLimit {
		if err := fscommon.WriteFile(path, strings.Join([]string{"hugetlb", hugetlb.Pagesize, "limit_in_bytes"}, "."), strconv.FormatUint(hugetlb.Limit, 10)); err != nil {
			return err
		}
	}

	return nil
}

func (s *HugetlbGroup) Remove(d *cgroupData) error {
	return removePath(d.path("hugetlb"))
}

func (s *HugetlbGroup) GetStats(path string, stats *cgroups.Stats) error {
	hugetlbStats := cgroups.HugetlbStats{}
	for _, pageSize := range HugePageSizes {
		usage := strings.Join([]string{"hugetlb", pageSize, "usage_in_bytes"}, ".")
		value, err := fscommon.GetCgroupParamUint(path, usage)
		if err != nil {
			return fmt.Errorf("failed to parse %s - %v", usage, err)
		}
		hugetlbStats.Usage = value

		maxUsage := strings.Join([]string{"hugetlb", pageSize, "max_usage_in_bytes"}, ".")
		value, err = fscommon.GetCgroupParamUint(path, maxUsage)
		if err != nil {
			return fmt.Errorf("failed to parse %s - %v", maxUsage, err)
		}
		hugetlbStats.MaxUsage = value

		failcnt := strings.Join([]string{"hugetlb", pageSize, "failcnt"}, ".")
		value, err = fscommon.GetCgroupParamUint(path, failcnt)
		if err != nil {
			return fmt.Errorf("failed to parse %s - %v", failcnt, err)
		}
		hugetlbStats.Failcnt = value

		stats.HugetlbStats[pageSize] = hugetlbStats
	}

	return nil
}
