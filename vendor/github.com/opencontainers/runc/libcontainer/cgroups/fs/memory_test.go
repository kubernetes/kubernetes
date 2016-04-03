// +build linux

package fs

import (
	"strconv"
	"testing"

	"github.com/opencontainers/runc/libcontainer/cgroups"
)

const (
	memoryStatContents = `cache 512
rss 1024`
	memoryUsageContents    = "2048\n"
	memoryMaxUsageContents = "4096\n"
	memoryFailcnt          = "100\n"
)

func TestMemorySetMemory(t *testing.T) {
	helper := NewCgroupTestUtil("memory", t)
	defer helper.cleanup()

	const (
		memoryBefore      = 314572800 // 300M
		memoryAfter       = 524288000 // 500M
		reservationBefore = 209715200 // 200M
		reservationAfter  = 314572800 // 300M
	)

	helper.writeFileContents(map[string]string{
		"memory.limit_in_bytes":      strconv.Itoa(memoryBefore),
		"memory.soft_limit_in_bytes": strconv.Itoa(reservationBefore),
	})

	helper.CgroupData.config.Resources.Memory = memoryAfter
	helper.CgroupData.config.Resources.MemoryReservation = reservationAfter
	memory := &MemoryGroup{}
	if err := memory.Set(helper.CgroupPath, helper.CgroupData.config); err != nil {
		t.Fatal(err)
	}

	value, err := getCgroupParamUint(helper.CgroupPath, "memory.limit_in_bytes")
	if err != nil {
		t.Fatalf("Failed to parse memory.limit_in_bytes - %s", err)
	}
	if value != memoryAfter {
		t.Fatal("Got the wrong value, set memory.limit_in_bytes failed.")
	}

	value, err = getCgroupParamUint(helper.CgroupPath, "memory.soft_limit_in_bytes")
	if err != nil {
		t.Fatalf("Failed to parse memory.soft_limit_in_bytes - %s", err)
	}
	if value != reservationAfter {
		t.Fatal("Got the wrong value, set memory.soft_limit_in_bytes failed.")
	}
}

func TestMemorySetMemoryswap(t *testing.T) {
	helper := NewCgroupTestUtil("memory", t)
	defer helper.cleanup()

	const (
		memoryswapBefore = 314572800 // 300M
		memoryswapAfter  = 524288000 // 500M
	)

	helper.writeFileContents(map[string]string{
		"memory.memsw.limit_in_bytes": strconv.Itoa(memoryswapBefore),
	})

	helper.CgroupData.config.Resources.MemorySwap = memoryswapAfter
	memory := &MemoryGroup{}
	if err := memory.Set(helper.CgroupPath, helper.CgroupData.config); err != nil {
		t.Fatal(err)
	}

	value, err := getCgroupParamUint(helper.CgroupPath, "memory.memsw.limit_in_bytes")
	if err != nil {
		t.Fatalf("Failed to parse memory.memsw.limit_in_bytes - %s", err)
	}
	if value != memoryswapAfter {
		t.Fatal("Got the wrong value, set memory.memsw.limit_in_bytes failed.")
	}
}

func TestMemorySetKernelMemory(t *testing.T) {
	helper := NewCgroupTestUtil("memory", t)
	defer helper.cleanup()

	const (
		kernelMemoryBefore = 314572800 // 300M
		kernelMemoryAfter  = 524288000 // 500M
	)

	helper.writeFileContents(map[string]string{
		"memory.kmem.limit_in_bytes": strconv.Itoa(kernelMemoryBefore),
	})

	helper.CgroupData.config.Resources.KernelMemory = kernelMemoryAfter
	memory := &MemoryGroup{}
	if err := memory.SetKernelMemory(helper.CgroupPath, helper.CgroupData.config); err != nil {
		t.Fatal(err)
	}

	value, err := getCgroupParamUint(helper.CgroupPath, "memory.kmem.limit_in_bytes")
	if err != nil {
		t.Fatalf("Failed to parse memory.kmem.limit_in_bytes - %s", err)
	}
	if value != kernelMemoryAfter {
		t.Fatal("Got the wrong value, set memory.kmem.limit_in_bytes failed.")
	}
}

func TestMemorySetMemorySwappinessDefault(t *testing.T) {
	helper := NewCgroupTestUtil("memory", t)
	defer helper.cleanup()

	const (
		swappinessBefore = 60 //deafult is 60
		swappinessAfter  = 0
	)

	helper.writeFileContents(map[string]string{
		"memory.swappiness": strconv.Itoa(swappinessBefore),
	})

	helper.CgroupData.config.Resources.Memory = swappinessAfter
	memory := &MemoryGroup{}
	if err := memory.Set(helper.CgroupPath, helper.CgroupData.config); err != nil {
		t.Fatal(err)
	}

	value, err := getCgroupParamUint(helper.CgroupPath, "memory.swappiness")
	if err != nil {
		t.Fatalf("Failed to parse memory.swappiness - %s", err)
	}
	if value != swappinessAfter {
		t.Fatal("Got the wrong value, set memory.swappiness failed.")
	}
}

func TestMemoryStats(t *testing.T) {
	helper := NewCgroupTestUtil("memory", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"memory.stat":                     memoryStatContents,
		"memory.usage_in_bytes":           memoryUsageContents,
		"memory.max_usage_in_bytes":       memoryMaxUsageContents,
		"memory.failcnt":                  memoryFailcnt,
		"memory.memsw.usage_in_bytes":     memoryUsageContents,
		"memory.memsw.max_usage_in_bytes": memoryMaxUsageContents,
		"memory.memsw.failcnt":            memoryFailcnt,
		"memory.kmem.usage_in_bytes":      memoryUsageContents,
		"memory.kmem.max_usage_in_bytes":  memoryMaxUsageContents,
		"memory.kmem.failcnt":             memoryFailcnt,
	})

	memory := &MemoryGroup{}
	actualStats := *cgroups.NewStats()
	err := memory.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatal(err)
	}
	expectedStats := cgroups.MemoryStats{Cache: 512, Usage: cgroups.MemoryData{Usage: 2048, MaxUsage: 4096, Failcnt: 100}, SwapUsage: cgroups.MemoryData{Usage: 2048, MaxUsage: 4096, Failcnt: 100}, KernelUsage: cgroups.MemoryData{Usage: 2048, MaxUsage: 4096, Failcnt: 100}, Stats: map[string]uint64{"cache": 512, "rss": 1024}}
	expectMemoryStatEquals(t, expectedStats, actualStats.MemoryStats)
}

func TestMemoryStatsNoStatFile(t *testing.T) {
	helper := NewCgroupTestUtil("memory", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"memory.usage_in_bytes":     memoryUsageContents,
		"memory.max_usage_in_bytes": memoryMaxUsageContents,
	})

	memory := &MemoryGroup{}
	actualStats := *cgroups.NewStats()
	err := memory.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatal(err)
	}
}

func TestMemoryStatsNoUsageFile(t *testing.T) {
	helper := NewCgroupTestUtil("memory", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"memory.stat":               memoryStatContents,
		"memory.max_usage_in_bytes": memoryMaxUsageContents,
	})

	memory := &MemoryGroup{}
	actualStats := *cgroups.NewStats()
	err := memory.GetStats(helper.CgroupPath, &actualStats)
	if err == nil {
		t.Fatal("Expected failure")
	}
}

func TestMemoryStatsNoMaxUsageFile(t *testing.T) {
	helper := NewCgroupTestUtil("memory", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"memory.stat":           memoryStatContents,
		"memory.usage_in_bytes": memoryUsageContents,
	})

	memory := &MemoryGroup{}
	actualStats := *cgroups.NewStats()
	err := memory.GetStats(helper.CgroupPath, &actualStats)
	if err == nil {
		t.Fatal("Expected failure")
	}
}

func TestMemoryStatsBadStatFile(t *testing.T) {
	helper := NewCgroupTestUtil("memory", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"memory.stat":               "rss rss",
		"memory.usage_in_bytes":     memoryUsageContents,
		"memory.max_usage_in_bytes": memoryMaxUsageContents,
	})

	memory := &MemoryGroup{}
	actualStats := *cgroups.NewStats()
	err := memory.GetStats(helper.CgroupPath, &actualStats)
	if err == nil {
		t.Fatal("Expected failure")
	}
}

func TestMemoryStatsBadUsageFile(t *testing.T) {
	helper := NewCgroupTestUtil("memory", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"memory.stat":               memoryStatContents,
		"memory.usage_in_bytes":     "bad",
		"memory.max_usage_in_bytes": memoryMaxUsageContents,
	})

	memory := &MemoryGroup{}
	actualStats := *cgroups.NewStats()
	err := memory.GetStats(helper.CgroupPath, &actualStats)
	if err == nil {
		t.Fatal("Expected failure")
	}
}

func TestMemoryStatsBadMaxUsageFile(t *testing.T) {
	helper := NewCgroupTestUtil("memory", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"memory.stat":               memoryStatContents,
		"memory.usage_in_bytes":     memoryUsageContents,
		"memory.max_usage_in_bytes": "bad",
	})

	memory := &MemoryGroup{}
	actualStats := *cgroups.NewStats()
	err := memory.GetStats(helper.CgroupPath, &actualStats)
	if err == nil {
		t.Fatal("Expected failure")
	}
}

func TestMemorySetOomControl(t *testing.T) {
	helper := NewCgroupTestUtil("memory", t)
	defer helper.cleanup()

	const (
		oom_kill_disable = 1 // disable oom killer, default is 0
	)

	helper.writeFileContents(map[string]string{
		"memory.oom_control": strconv.Itoa(oom_kill_disable),
	})

	memory := &MemoryGroup{}
	if err := memory.Set(helper.CgroupPath, helper.CgroupData.config); err != nil {
		t.Fatal(err)
	}

	value, err := getCgroupParamUint(helper.CgroupPath, "memory.oom_control")
	if err != nil {
		t.Fatalf("Failed to parse memory.oom_control - %s", err)
	}

	if value != oom_kill_disable {
		t.Fatalf("Got the wrong value, set memory.oom_control failed.")
	}
}
