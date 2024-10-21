package fs2

import (
	"errors"
	"os"
	"strconv"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

func isHugeTlbSet(r *configs.Resources) bool {
	return len(r.HugetlbLimit) > 0
}

func setHugeTlb(dirPath string, r *configs.Resources) error {
	if !isHugeTlbSet(r) {
		return nil
	}
	const suffix = ".max"
	skipRsvd := false
	for _, hugetlb := range r.HugetlbLimit {
		prefix := "hugetlb." + hugetlb.Pagesize
		val := strconv.FormatUint(hugetlb.Limit, 10)
		if err := cgroups.WriteFile(dirPath, prefix+suffix, val); err != nil {
			return err
		}
		if skipRsvd {
			continue
		}
		if err := cgroups.WriteFile(dirPath, prefix+".rsvd"+suffix, val); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				skipRsvd = true
				continue
			}
			return err
		}
	}

	return nil
}

func statHugeTlb(dirPath string, stats *cgroups.Stats) error {
	hugetlbStats := cgroups.HugetlbStats{}
	rsvd := ".rsvd"
	for _, pagesize := range cgroups.HugePageSizes() {
	again:
		prefix := "hugetlb." + pagesize + rsvd
		value, err := fscommon.GetCgroupParamUint(dirPath, prefix+".current")
		if err != nil {
			if rsvd != "" && errors.Is(err, os.ErrNotExist) {
				rsvd = ""
				goto again
			}
			return err
		}
		hugetlbStats.Usage = value

		value, err = fscommon.GetValueByKey(dirPath, prefix+".events", "max")
		if err != nil {
			return err
		}
		hugetlbStats.Failcnt = value

		stats.HugetlbStats[pagesize] = hugetlbStats
	}

	return nil
}
