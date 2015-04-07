package fs

import (
	"fmt"
	"testing"

	"github.com/docker/libcontainer/cgroups"
)

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
