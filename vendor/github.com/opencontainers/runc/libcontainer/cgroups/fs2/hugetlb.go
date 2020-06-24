// +build linux

package fs2

import (
	"io/ioutil"
	"path/filepath"
	"strconv"
	"strings"

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
		if err := fscommon.WriteFile(dirPath, strings.Join([]string{"hugetlb", hugetlb.Pagesize, "max"}, "."), strconv.FormatUint(hugetlb.Limit, 10)); err != nil {
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
		usage := strings.Join([]string{"hugetlb", pagesize, "current"}, ".")
		value, err := fscommon.GetCgroupParamUint(dirPath, usage)
		if err != nil {
			return errors.Wrapf(err, "failed to parse hugetlb.%s.current file", pagesize)
		}
		hugetlbStats.Usage = value

		fileName := strings.Join([]string{"hugetlb", pagesize, "events"}, ".")
		filePath := filepath.Join(dirPath, fileName)
		contents, err := ioutil.ReadFile(filePath)
		if err != nil {
			return errors.Wrapf(err, "failed to parse hugetlb.%s.events file", pagesize)
		}
		_, value, err = fscommon.GetCgroupParamKeyValue(string(contents))
		if err != nil {
			return errors.Wrapf(err, "failed to parse hugetlb.%s.events file", pagesize)
		}
		hugetlbStats.Failcnt = value

		stats.HugetlbStats[pagesize] = hugetlbStats
	}

	return nil
}
