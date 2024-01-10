package fs

import (
	"errors"
	"os"
	"strconv"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type HugetlbGroup struct{}

func (s *HugetlbGroup) Name() string {
	return "hugetlb"
}

func (s *HugetlbGroup) Apply(path string, _ *configs.Resources, pid int) error {
	return apply(path, pid)
}

func (s *HugetlbGroup) Set(path string, r *configs.Resources) error {
	const suffix = ".limit_in_bytes"
	skipRsvd := false

	for _, hugetlb := range r.HugetlbLimit {
		prefix := "hugetlb." + hugetlb.Pagesize
		val := strconv.FormatUint(hugetlb.Limit, 10)
		if err := cgroups.WriteFile(path, prefix+suffix, val); err != nil {
			return err
		}
		if skipRsvd {
			continue
		}
		if err := cgroups.WriteFile(path, prefix+".rsvd"+suffix, val); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				skipRsvd = true
				continue
			}
			return err
		}
	}

	return nil
}

func (s *HugetlbGroup) GetStats(path string, stats *cgroups.Stats) error {
	if !cgroups.PathExists(path) {
		return nil
	}
	rsvd := ".rsvd"
	hugetlbStats := cgroups.HugetlbStats{}
	for _, pageSize := range cgroups.HugePageSizes() {
	again:
		prefix := "hugetlb." + pageSize + rsvd

		value, err := fscommon.GetCgroupParamUint(path, prefix+".usage_in_bytes")
		if err != nil {
			if rsvd != "" && errors.Is(err, os.ErrNotExist) {
				rsvd = ""
				goto again
			}
			return err
		}
		hugetlbStats.Usage = value

		value, err = fscommon.GetCgroupParamUint(path, prefix+".max_usage_in_bytes")
		if err != nil {
			return err
		}
		hugetlbStats.MaxUsage = value

		value, err = fscommon.GetCgroupParamUint(path, prefix+".failcnt")
		if err != nil {
			return err
		}
		hugetlbStats.Failcnt = value

		stats.HugetlbStats[pageSize] = hugetlbStats
	}

	return nil
}
