// +build linux

package fs2

import (
	"strconv"

	"github.com/pkg/errors"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
)

func isHugeTlbSet(cgroup *configs.Cgroup) bool {
	return len(cgroup.Resources.HugetlbLimit) > 0
}

func setHugeTlb(dirPath string, cgroup *configs.Cgroup) error {
	if !isHugeTlbSet(cgroup) {
		return nil
	}
	for _, hugetlb := range cgroup.Resources.HugetlbLimit {
		if err := fscommon.WriteFile(dirPath, "hugetlb."+hugetlb.Pagesize+".max", strconv.FormatUint(hugetlb.Limit, 10)); err != nil {
			return err
		}
	}

	return nil
}

func statHugeTlb(dirPath string, stats *cgroups.Stats) error {
	hugePageSizes, err := cgroups.GetHugePageSize()
	if err != nil {
		return errors.Wrap(err, "failed to fetch hugetlb info")
	}
	hugetlbStats := cgroups.HugetlbStats{}

	for _, pagesize := range hugePageSizes {
		value, err := fscommon.GetCgroupParamUint(dirPath, "hugetlb."+pagesize+".current")
		if err != nil {
			return err
		}
		hugetlbStats.Usage = value

		fileName := "hugetlb." + pagesize + ".events"
		contents, err := fscommon.ReadFile(dirPath, fileName)
		if err != nil {
			return errors.Wrap(err, "failed to read stats")
		}
		_, value, err = fscommon.GetCgroupParamKeyValue(contents)
		if err != nil {
			return errors.Wrap(err, "failed to parse "+fileName)
		}
		hugetlbStats.Failcnt = value

		stats.HugetlbStats[pagesize] = hugetlbStats
	}

	return nil
}
