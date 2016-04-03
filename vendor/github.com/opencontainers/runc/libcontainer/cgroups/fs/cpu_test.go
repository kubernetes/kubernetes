// +build linux

package fs

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/opencontainers/runc/libcontainer/cgroups"
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

	helper.CgroupData.config.Resources.CpuShares = sharesAfter
	cpu := &CpuGroup{}
	if err := cpu.Set(helper.CgroupPath, helper.CgroupData.config); err != nil {
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
		quotaBefore     = 8000
		quotaAfter      = 5000
		periodBefore    = 10000
		periodAfter     = 7000
		rtRuntimeBefore = 8000
		rtRuntimeAfter  = 5000
		rtPeriodBefore  = 10000
		rtPeriodAfter   = 7000
	)

	helper.writeFileContents(map[string]string{
		"cpu.cfs_quota_us":  strconv.Itoa(quotaBefore),
		"cpu.cfs_period_us": strconv.Itoa(periodBefore),
		"cpu.rt_runtime_us": strconv.Itoa(rtRuntimeBefore),
		"cpu.rt_period_us":  strconv.Itoa(rtPeriodBefore),
	})

	helper.CgroupData.config.Resources.CpuQuota = quotaAfter
	helper.CgroupData.config.Resources.CpuPeriod = periodAfter
	helper.CgroupData.config.Resources.CpuRtRuntime = rtRuntimeAfter
	helper.CgroupData.config.Resources.CpuRtPeriod = rtPeriodAfter
	cpu := &CpuGroup{}
	if err := cpu.Set(helper.CgroupPath, helper.CgroupData.config); err != nil {
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
	rtRuntime, err := getCgroupParamUint(helper.CgroupPath, "cpu.rt_runtime_us")
	if err != nil {
		t.Fatalf("Failed to parse cpu.rt_runtime_us - %s", err)
	}
	if rtRuntime != rtRuntimeAfter {
		t.Fatal("Got the wrong value, set cpu.rt_runtime_us failed.")
	}
	rtPeriod, err := getCgroupParamUint(helper.CgroupPath, "cpu.rt_period_us")
	if err != nil {
		t.Fatalf("Failed to parse cpu.rt_period_us - %s", err)
	}
	if rtPeriod != rtPeriodAfter {
		t.Fatal("Got the wrong value, set cpu.rt_period_us failed.")
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
