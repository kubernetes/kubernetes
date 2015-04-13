package fs

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/docker/libcontainer/cgroups"
)

func TestCpuSetShares(t *testing.T) {
	helper := NewCgroupTestUtil("cpu", t)
	defer helper.cleanup()

	const (
		sharesBefore = 1024
		sharesAfter  = 512
	)

	helper.writeFileContents(map[string]string{
		"cpu.shares": strconv.Itoa(sharesBefore),
	})

	helper.CgroupData.c.CpuShares = sharesAfter
	cpu := &CpuGroup{}
	if err := cpu.Set(helper.CgroupPath, helper.CgroupData.c); err != nil {
		t.Fatal(err)
	}

	value, err := getCgroupParamUint(helper.CgroupPath, "cpu.shares")
	if err != nil {
		t.Fatalf("Failed to parse cpu.shares - %s", err)
	}

	if value != sharesAfter {
		t.Fatal("Got the wrong value, set cpu.shares failed.")
	}
}

func TestCpuSetBandWidth(t *testing.T) {
	helper := NewCgroupTestUtil("cpu", t)
	defer helper.cleanup()

	const (
		quotaBefore  = 8000
		quotaAfter   = 5000
		periodBefore = 10000
		periodAfter  = 7000
	)

	helper.writeFileContents(map[string]string{
		"cpu.cfs_quota_us":  strconv.Itoa(quotaBefore),
		"cpu.cfs_period_us": strconv.Itoa(periodBefore),
	})

	helper.CgroupData.c.CpuQuota = quotaAfter
	helper.CgroupData.c.CpuPeriod = periodAfter
	cpu := &CpuGroup{}
	if err := cpu.Set(helper.CgroupPath, helper.CgroupData.c); err != nil {
		t.Fatal(err)
	}

	quota, err := getCgroupParamUint(helper.CgroupPath, "cpu.cfs_quota_us")
	if err != nil {
		t.Fatalf("Failed to parse cpu.cfs_quota_us - %s", err)
	}
	if quota != quotaAfter {
		t.Fatal("Got the wrong value, set cpu.cfs_quota_us failed.")
	}

	period, err := getCgroupParamUint(helper.CgroupPath, "cpu.cfs_period_us")
	if err != nil {
		t.Fatalf("Failed to parse cpu.cfs_period_us - %s", err)
	}
	if period != periodAfter {
		t.Fatal("Got the wrong value, set cpu.cfs_period_us failed.")
	}
}

func TestCpuStats(t *testing.T) {
	helper := NewCgroupTestUtil("cpu", t)
	defer helper.cleanup()

	const (
		kNrPeriods     = 2000
		kNrThrottled   = 200
		kThrottledTime = uint64(18446744073709551615)
	)

	cpuStatContent := fmt.Sprintf("nr_periods %d\n nr_throttled %d\n throttled_time %d\n",
		kNrPeriods, kNrThrottled, kThrottledTime)
	helper.writeFileContents(map[string]string{
		"cpu.stat": cpuStatContent,
	})

	cpu := &CpuGroup{}
	actualStats := *cgroups.NewStats()
	err := cpu.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatal(err)
	}

	expectedStats := cgroups.ThrottlingData{
		Periods:          kNrPeriods,
		ThrottledPeriods: kNrThrottled,
		ThrottledTime:    kThrottledTime}

	expectThrottlingDataEquals(t, expectedStats, actualStats.CpuStats.ThrottlingData)
}

func TestNoCpuStatFile(t *testing.T) {
	helper := NewCgroupTestUtil("cpu", t)
	defer helper.cleanup()

	cpu := &CpuGroup{}
	actualStats := *cgroups.NewStats()
	err := cpu.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatal("Expected not to fail, but did")
	}
}

func TestInvalidCpuStat(t *testing.T) {
	helper := NewCgroupTestUtil("cpu", t)
	defer helper.cleanup()
	cpuStatContent := `nr_periods 2000
	nr_throttled 200
	throttled_time fortytwo`
	helper.writeFileContents(map[string]string{
		"cpu.stat": cpuStatContent,
	})

	cpu := &CpuGroup{}
	actualStats := *cgroups.NewStats()
	err := cpu.GetStats(helper.CgroupPath, &actualStats)
	if err == nil {
		t.Fatal("Expected failed stat parsing.")
	}
}
