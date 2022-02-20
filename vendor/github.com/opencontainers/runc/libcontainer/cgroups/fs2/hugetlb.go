// +build linux

package fs2

import (
	"strconv"

	"github.com/pkg/errors"

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
	for _, hugetlb := range r.HugetlbLimit {
		if err := cgroups.WriteFile(dirPath, "hugetlb."+hugetlb.Pagesize+".max", strconv.FormatUint(hugetlb.Limit, 10)); err != nil {
			return err
		}
	}

	return nil
}

func statHugeTlb(dirPath string, stats *cgroups.Stats) error {
	hugePageSizes, _ := cgroups.GetHugePageSize()
	hugetlbStats := cgroups.HugetlbStats{}

	for _, pagesize := range hugePageSizes {
		value, err := fscommon.GetCgroupParamUint(dirPath, "hugetlb."+pagesize+".current")
		if err != nil {
			return err
		}
		hugetlbStats.Usage = value

		fileName := "hugetlb." + pagesize + ".events"
		value, err = fscommon.GetValueByKey(dirPath, fileName, "max")
		if err != nil {
			return errors.Wrap(err, "failed to read stats")
		}
		hugetlbStats.Failcnt = value

		stats.HugetlbStats[pagesize] = hugetlbStats
	}

	return nil
}
