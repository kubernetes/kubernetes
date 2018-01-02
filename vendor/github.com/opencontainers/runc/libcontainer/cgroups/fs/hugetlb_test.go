// +build linux

package fs

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
)

const (
	hugetlbUsageContents    = "128\n"
	hugetlbMaxUsageContents = "256\n"
	hugetlbFailcnt          = "100\n"
)

var (
	usage    = "hugetlb.%s.usage_in_bytes"
	limit    = "hugetlb.%s.limit_in_bytes"
	maxUsage = "hugetlb.%s.max_usage_in_bytes"
	failcnt  = "hugetlb.%s.failcnt"
)

func TestHugetlbSetHugetlb(t *testing.T) {
	helper := NewCgroupTestUtil("hugetlb", t)
	defer helper.cleanup()

	const (
		hugetlbBefore = 256
		hugetlbAfter  = 512
	)

	for _, pageSize := range HugePageSizes {
		helper.writeFileContents(map[string]string{
			fmt.Sprintf(limit, pageSize): strconv.Itoa(hugetlbBefore),
		})
	}

	for _, pageSize := range HugePageSizes {
		helper.CgroupData.config.Resources.HugetlbLimit = []*configs.HugepageLimit{
			{
				Pagesize: pageSize,
				Limit:    hugetlbAfter,
			},
		}
		hugetlb := &HugetlbGroup{}
		if err := hugetlb.Set(helper.CgroupPath, helper.CgroupData.config); err != nil {
			t.Fatal(err)
		}
	}

	for _, pageSize := range HugePageSizes {
		limit := fmt.Sprintf(limit, pageSize)
		value, err := getCgroupParamUint(helper.CgroupPath, limit)
		if err != nil {
			t.Fatalf("Failed to parse %s - %s", limit, err)
		}
		if value != hugetlbAfter {
			t.Fatalf("Set hugetlb.limit_in_bytes failed. Expected: %v, Got: %v", hugetlbAfter, value)
		}
	}
}

func TestHugetlbStats(t *testing.T) {
	helper := NewCgroupTestUtil("hugetlb", t)
	defer helper.cleanup()
	for _, pageSize := range HugePageSizes {
		helper.writeFileContents(map[string]string{
			fmt.Sprintf(usage, pageSize):    hugetlbUsageContents,
			fmt.Sprintf(maxUsage, pageSize): hugetlbMaxUsageContents,
			fmt.Sprintf(failcnt, pageSize):  hugetlbFailcnt,
		})
	}

	hugetlb := &HugetlbGroup{}
	actualStats := *cgroups.NewStats()
	err := hugetlb.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatal(err)
	}
	expectedStats := cgroups.HugetlbStats{Usage: 128, MaxUsage: 256, Failcnt: 100}
	for _, pageSize := range HugePageSizes {
		expectHugetlbStatEquals(t, expectedStats, actualStats.HugetlbStats[pageSize])
	}
}

func TestHugetlbStatsNoUsageFile(t *testing.T) {
	helper := NewCgroupTestUtil("hugetlb", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		maxUsage: hugetlbMaxUsageContents,
	})

	hugetlb := &HugetlbGroup{}
	actualStats := *cgroups.NewStats()
	err := hugetlb.GetStats(helper.CgroupPath, &actualStats)
	if err == nil {
		t.Fatal("Expected failure")
	}
}

func TestHugetlbStatsNoMaxUsageFile(t *testing.T) {
	helper := NewCgroupTestUtil("hugetlb", t)
	defer helper.cleanup()
	for _, pageSize := range HugePageSizes {
		helper.writeFileContents(map[string]string{
			fmt.Sprintf(usage, pageSize): hugetlbUsageContents,
		})
	}

	hugetlb := &HugetlbGroup{}
	actualStats := *cgroups.NewStats()
	err := hugetlb.GetStats(helper.CgroupPath, &actualStats)
	if err == nil {
		t.Fatal("Expected failure")
	}
}

func TestHugetlbStatsBadUsageFile(t *testing.T) {
	helper := NewCgroupTestUtil("hugetlb", t)
	defer helper.cleanup()
	for _, pageSize := range HugePageSizes {
		helper.writeFileContents(map[string]string{
			fmt.Sprintf(usage, pageSize): "bad",
			maxUsage:                     hugetlbMaxUsageContents,
		})
	}

	hugetlb := &HugetlbGroup{}
	actualStats := *cgroups.NewStats()
	err := hugetlb.GetStats(helper.CgroupPath, &actualStats)
	if err == nil {
		t.Fatal("Expected failure")
	}
}

func TestHugetlbStatsBadMaxUsageFile(t *testing.T) {
	helper := NewCgroupTestUtil("hugetlb", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		usage:    hugetlbUsageContents,
		maxUsage: "bad",
	})

	hugetlb := &HugetlbGroup{}
	actualStats := *cgroups.NewStats()
	err := hugetlb.GetStats(helper.CgroupPath, &actualStats)
	if err == nil {
		t.Fatal("Expected failure")
	}
}
