package fs

import (
	"strconv"
	"testing"

	"github.com/docker/libcontainer/cgroups"
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

	helper.CgroupData.c.Memory = memoryAfter
	helper.CgroupData.c.MemoryReservation = reservationAfter
	memory := &MemoryGroup{}
	if err := memory.Set(helper.CgroupPath, helper.CgroupData.c); err != nil {
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

	helper.CgroupData.c.MemorySwap = memoryswapAfter
	memory := &MemoryGroup{}
	if err := memory.Set(helper.CgroupPath, helper.CgroupData.c); err != nil {
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

func TestMemorySetMemoryswapDefault(t *testing.T) {
	helper := NewCgroupTestUtil("memory", t)
	defer helper.cleanup()

	const (
		memoryBefore    = 209715200 // 200M
		memoryAfter     = 314572800 // 300M
		memoryswapAfter = 629145600 // 300M*2
	)

	helper.writeFileContents(map[string]string{
		"memory.limit_in_bytes": strconv.Itoa(memoryBefore),
	})

	helper.CgroupData.c.Memory = memoryAfter
	memory := &MemoryGroup{}
	if err := memory.Set(helper.CgroupPath, helper.CgroupData.c); err != nil {
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

func TestMemoryStats(t *testing.T) {
	helper := NewCgroupTestUtil("memory", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"memory.stat":               memoryStatContents,
		"memory.usage_in_bytes":     memoryUsageContents,
		"memory.max_usage_in_bytes": memoryMaxUsageContents,
		"memory.failcnt":            memoryFailcnt,
	})

	memory := &MemoryGroup{}
	actualStats := *cgroups.NewStats()
	err := memory.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatal(err)
	}
	expectedStats := cgroups.MemoryStats{Usage: 2048, Cache: 512, MaxUsage: 4096, Failcnt: 100, Stats: map[string]uint64{"cache": 512, "rss": 1024}}
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
	if err := memory.Set(helper.CgroupPath, helper.CgroupData.c); err != nil {
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
