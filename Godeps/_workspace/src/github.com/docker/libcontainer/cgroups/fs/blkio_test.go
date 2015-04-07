package fs

import (
	"testing"

	"github.com/docker/libcontainer/cgroups"
)

const (
	sectorsRecursiveContents      = `8:0 1024`
	serviceBytesRecursiveContents = `8:0 Read 100
8:0 Write 200
8:0 Sync 300
8:0 Async 500
8:0 Total 500
Total 500`
	servicedRecursiveContents = `8:0 Read 10
8:0 Write 40
8:0 Sync 20
8:0 Async 30
8:0 Total 50
Total 50`
	queuedRecursiveContents = `8:0 Read 1
8:0 Write 4
8:0 Sync 2
8:0 Async 3
8:0 Total 5
Total 5`
	serviceTimeRecursiveContents = `8:0 Read 173959
8:0 Write 0
8:0 Sync 0
8:0 Async 173959
8:0 Total 17395
Total 17395`
	waitTimeRecursiveContents = `8:0 Read 15571
8:0 Write 0
8:0 Sync 0
8:0 Async 15571
8:0 Total 15571`
	mergedRecursiveContents = `8:0 Read 5
8:0 Write 10
8:0 Sync 0
8:0 Async 0
8:0 Total 15
Total 15`
	timeRecursiveContents = `8:0 8`
	throttleServiceBytes  = `8:0 Read 11030528
8:0 Write 23
8:0 Sync 42
8:0 Async 11030528
8:0 Total 11030528
252:0 Read 11030528
252:0 Write 23
252:0 Sync 42
252:0 Async 11030528
252:0 Total 11030528
Total 22061056`
	throttleServiced = `8:0 Read 164
8:0 Write 23
8:0 Sync 42
8:0 Async 164
8:0 Total 164
252:0 Read 164
252:0 Write 23
252:0 Sync 42
252:0 Async 164
252:0 Total 164
Total 328`
)

func appendBlkioStatEntry(blkioStatEntries *[]cgroups.BlkioStatEntry, major, minor, value uint64, op string) {
	*blkioStatEntries = append(*blkioStatEntries, cgroups.BlkioStatEntry{Major: major, Minor: minor, Value: value, Op: op})
}

func TestBlkioStats(t *testing.T) {
	helper := NewCgroupTestUtil("blkio", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"blkio.io_service_bytes_recursive": serviceBytesRecursiveContents,
		"blkio.io_serviced_recursive":      servicedRecursiveContents,
		"blkio.io_queued_recursive":        queuedRecursiveContents,
		"blkio.io_service_time_recursive":  serviceTimeRecursiveContents,
		"blkio.io_wait_time_recursive":     waitTimeRecursiveContents,
		"blkio.io_merged_recursive":        mergedRecursiveContents,
		"blkio.time_recursive":             timeRecursiveContents,
		"blkio.sectors_recursive":          sectorsRecursiveContents,
	})

	blkio := &BlkioGroup{}
	actualStats := *cgroups.NewStats()
	err := blkio.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatal(err)
	}

	// Verify expected stats.
	expectedStats := cgroups.BlkioStats{}
	appendBlkioStatEntry(&expectedStats.SectorsRecursive, 8, 0, 1024, "")

	appendBlkioStatEntry(&expectedStats.IoServiceBytesRecursive, 8, 0, 100, "Read")
	appendBlkioStatEntry(&expectedStats.IoServiceBytesRecursive, 8, 0, 200, "Write")
	appendBlkioStatEntry(&expectedStats.IoServiceBytesRecursive, 8, 0, 300, "Sync")
	appendBlkioStatEntry(&expectedStats.IoServiceBytesRecursive, 8, 0, 500, "Async")
	appendBlkioStatEntry(&expectedStats.IoServiceBytesRecursive, 8, 0, 500, "Total")

	appendBlkioStatEntry(&expectedStats.IoServicedRecursive, 8, 0, 10, "Read")
	appendBlkioStatEntry(&expectedStats.IoServicedRecursive, 8, 0, 40, "Write")
	appendBlkioStatEntry(&expectedStats.IoServicedRecursive, 8, 0, 20, "Sync")
	appendBlkioStatEntry(&expectedStats.IoServicedRecursive, 8, 0, 30, "Async")
	appendBlkioStatEntry(&expectedStats.IoServicedRecursive, 8, 0, 50, "Total")

	appendBlkioStatEntry(&expectedStats.IoQueuedRecursive, 8, 0, 1, "Read")
	appendBlkioStatEntry(&expectedStats.IoQueuedRecursive, 8, 0, 4, "Write")
	appendBlkioStatEntry(&expectedStats.IoQueuedRecursive, 8, 0, 2, "Sync")
	appendBlkioStatEntry(&expectedStats.IoQueuedRecursive, 8, 0, 3, "Async")
	appendBlkioStatEntry(&expectedStats.IoQueuedRecursive, 8, 0, 5, "Total")

	appendBlkioStatEntry(&expectedStats.IoServiceTimeRecursive, 8, 0, 173959, "Read")
	appendBlkioStatEntry(&expectedStats.IoServiceTimeRecursive, 8, 0, 0, "Write")
	appendBlkioStatEntry(&expectedStats.IoServiceTimeRecursive, 8, 0, 0, "Sync")
	appendBlkioStatEntry(&expectedStats.IoServiceTimeRecursive, 8, 0, 173959, "Async")
	appendBlkioStatEntry(&expectedStats.IoServiceTimeRecursive, 8, 0, 17395, "Total")

	appendBlkioStatEntry(&expectedStats.IoWaitTimeRecursive, 8, 0, 15571, "Read")
	appendBlkioStatEntry(&expectedStats.IoWaitTimeRecursive, 8, 0, 0, "Write")
	appendBlkioStatEntry(&expectedStats.IoWaitTimeRecursive, 8, 0, 0, "Sync")
	appendBlkioStatEntry(&expectedStats.IoWaitTimeRecursive, 8, 0, 15571, "Async")
	appendBlkioStatEntry(&expectedStats.IoWaitTimeRecursive, 8, 0, 15571, "Total")

	appendBlkioStatEntry(&expectedStats.IoMergedRecursive, 8, 0, 5, "Read")
	appendBlkioStatEntry(&expectedStats.IoMergedRecursive, 8, 0, 10, "Write")
	appendBlkioStatEntry(&expectedStats.IoMergedRecursive, 8, 0, 0, "Sync")
	appendBlkioStatEntry(&expectedStats.IoMergedRecursive, 8, 0, 0, "Async")
	appendBlkioStatEntry(&expectedStats.IoMergedRecursive, 8, 0, 15, "Total")

	appendBlkioStatEntry(&expectedStats.IoTimeRecursive, 8, 0, 8, "")

	expectBlkioStatsEquals(t, expectedStats, actualStats.BlkioStats)
}

func TestBlkioStatsNoSectorsFile(t *testing.T) {
	helper := NewCgroupTestUtil("blkio", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"blkio.io_service_bytes_recursive": serviceBytesRecursiveContents,
		"blkio.io_serviced_recursive":      servicedRecursiveContents,
		"blkio.io_queued_recursive":        queuedRecursiveContents,
		"blkio.io_service_time_recursive":  serviceTimeRecursiveContents,
		"blkio.io_wait_time_recursive":     waitTimeRecursiveContents,
		"blkio.io_merged_recursive":        mergedRecursiveContents,
		"blkio.time_recursive":             timeRecursiveContents,
	})

	blkio := &BlkioGroup{}
	actualStats := *cgroups.NewStats()
	err := blkio.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatalf("Failed unexpectedly: %s", err)
	}
}

func TestBlkioStatsNoServiceBytesFile(t *testing.T) {
	helper := NewCgroupTestUtil("blkio", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"blkio.io_serviced_recursive":     servicedRecursiveContents,
		"blkio.io_queued_recursive":       queuedRecursiveContents,
		"blkio.sectors_recursive":         sectorsRecursiveContents,
		"blkio.io_service_time_recursive": serviceTimeRecursiveContents,
		"blkio.io_wait_time_recursive":    waitTimeRecursiveContents,
		"blkio.io_merged_recursive":       mergedRecursiveContents,
		"blkio.time_recursive":            timeRecursiveContents,
	})

	blkio := &BlkioGroup{}
	actualStats := *cgroups.NewStats()
	err := blkio.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatalf("Failed unexpectedly: %s", err)
	}
}

func TestBlkioStatsNoServicedFile(t *testing.T) {
	helper := NewCgroupTestUtil("blkio", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"blkio.io_service_bytes_recursive": serviceBytesRecursiveContents,
		"blkio.io_queued_recursive":        queuedRecursiveContents,
		"blkio.sectors_recursive":          sectorsRecursiveContents,
		"blkio.io_service_time_recursive":  serviceTimeRecursiveContents,
		"blkio.io_wait_time_recursive":     waitTimeRecursiveContents,
		"blkio.io_merged_recursive":        mergedRecursiveContents,
		"blkio.time_recursive":             timeRecursiveContents,
	})

	blkio := &BlkioGroup{}
	actualStats := *cgroups.NewStats()
	err := blkio.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatalf("Failed unexpectedly: %s", err)
	}
}

func TestBlkioStatsNoQueuedFile(t *testing.T) {
	helper := NewCgroupTestUtil("blkio", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"blkio.io_service_bytes_recursive": serviceBytesRecursiveContents,
		"blkio.io_serviced_recursive":      servicedRecursiveContents,
		"blkio.sectors_recursive":          sectorsRecursiveContents,
		"blkio.io_service_time_recursive":  serviceTimeRecursiveContents,
		"blkio.io_wait_time_recursive":     waitTimeRecursiveContents,
		"blkio.io_merged_recursive":        mergedRecursiveContents,
		"blkio.time_recursive":             timeRecursiveContents,
	})

	blkio := &BlkioGroup{}
	actualStats := *cgroups.NewStats()
	err := blkio.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatalf("Failed unexpectedly: %s", err)
	}
}

func TestBlkioStatsNoServiceTimeFile(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	helper := NewCgroupTestUtil("blkio", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"blkio.io_service_bytes_recursive": serviceBytesRecursiveContents,
		"blkio.io_serviced_recursive":      servicedRecursiveContents,
		"blkio.io_queued_recursive":        queuedRecursiveContents,
		"blkio.io_wait_time_recursive":     waitTimeRecursiveContents,
		"blkio.io_merged_recursive":        mergedRecursiveContents,
		"blkio.time_recursive":             timeRecursiveContents,
		"blkio.sectors_recursive":          sectorsRecursiveContents,
	})

	blkio := &BlkioGroup{}
	actualStats := *cgroups.NewStats()
	err := blkio.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatalf("Failed unexpectedly: %s", err)
	}
}

func TestBlkioStatsNoWaitTimeFile(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	helper := NewCgroupTestUtil("blkio", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"blkio.io_service_bytes_recursive": serviceBytesRecursiveContents,
		"blkio.io_serviced_recursive":      servicedRecursiveContents,
		"blkio.io_queued_recursive":        queuedRecursiveContents,
		"blkio.io_service_time_recursive":  serviceTimeRecursiveContents,
		"blkio.io_merged_recursive":        mergedRecursiveContents,
		"blkio.time_recursive":             timeRecursiveContents,
		"blkio.sectors_recursive":          sectorsRecursiveContents,
	})

	blkio := &BlkioGroup{}
	actualStats := *cgroups.NewStats()
	err := blkio.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatalf("Failed unexpectedly: %s", err)
	}
}

func TestBlkioStatsNoMergedFile(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	helper := NewCgroupTestUtil("blkio", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"blkio.io_service_bytes_recursive": serviceBytesRecursiveContents,
		"blkio.io_serviced_recursive":      servicedRecursiveContents,
		"blkio.io_queued_recursive":        queuedRecursiveContents,
		"blkio.io_service_time_recursive":  serviceTimeRecursiveContents,
		"blkio.io_wait_time_recursive":     waitTimeRecursiveContents,
		"blkio.time_recursive":             timeRecursiveContents,
		"blkio.sectors_recursive":          sectorsRecursiveContents,
	})

	blkio := &BlkioGroup{}
	actualStats := *cgroups.NewStats()
	err := blkio.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatalf("Failed unexpectedly: %s", err)
	}
}

func TestBlkioStatsNoTimeFile(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	helper := NewCgroupTestUtil("blkio", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"blkio.io_service_bytes_recursive": serviceBytesRecursiveContents,
		"blkio.io_serviced_recursive":      servicedRecursiveContents,
		"blkio.io_queued_recursive":        queuedRecursiveContents,
		"blkio.io_service_time_recursive":  serviceTimeRecursiveContents,
		"blkio.io_wait_time_recursive":     waitTimeRecursiveContents,
		"blkio.io_merged_recursive":        mergedRecursiveContents,
		"blkio.sectors_recursive":          sectorsRecursiveContents,
	})

	blkio := &BlkioGroup{}
	actualStats := *cgroups.NewStats()
	err := blkio.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatalf("Failed unexpectedly: %s", err)
	}
}

func TestBlkioStatsUnexpectedNumberOfFields(t *testing.T) {
	helper := NewCgroupTestUtil("blkio", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"blkio.io_service_bytes_recursive": "8:0 Read 100 100",
		"blkio.io_serviced_recursive":      servicedRecursiveContents,
		"blkio.io_queued_recursive":        queuedRecursiveContents,
		"blkio.sectors_recursive":          sectorsRecursiveContents,
		"blkio.io_service_time_recursive":  serviceTimeRecursiveContents,
		"blkio.io_wait_time_recursive":     waitTimeRecursiveContents,
		"blkio.io_merged_recursive":        mergedRecursiveContents,
		"blkio.time_recursive":             timeRecursiveContents,
	})

	blkio := &BlkioGroup{}
	actualStats := *cgroups.NewStats()
	err := blkio.GetStats(helper.CgroupPath, &actualStats)
	if err == nil {
		t.Fatal("Expected to fail, but did not")
	}
}

func TestBlkioStatsUnexpectedFieldType(t *testing.T) {
	helper := NewCgroupTestUtil("blkio", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"blkio.io_service_bytes_recursive": "8:0 Read Write",
		"blkio.io_serviced_recursive":      servicedRecursiveContents,
		"blkio.io_queued_recursive":        queuedRecursiveContents,
		"blkio.sectors_recursive":          sectorsRecursiveContents,
		"blkio.io_service_time_recursive":  serviceTimeRecursiveContents,
		"blkio.io_wait_time_recursive":     waitTimeRecursiveContents,
		"blkio.io_merged_recursive":        mergedRecursiveContents,
		"blkio.time_recursive":             timeRecursiveContents,
	})

	blkio := &BlkioGroup{}
	actualStats := *cgroups.NewStats()
	err := blkio.GetStats(helper.CgroupPath, &actualStats)
	if err == nil {
		t.Fatal("Expected to fail, but did not")
	}
}

func TestNonCFQBlkioStats(t *testing.T) {
	helper := NewCgroupTestUtil("blkio", t)
	defer helper.cleanup()
	helper.writeFileContents(map[string]string{
		"blkio.io_service_bytes_recursive": "",
		"blkio.io_serviced_recursive":      "",
		"blkio.io_queued_recursive":        "",
		"blkio.sectors_recursive":          "",
		"blkio.io_service_time_recursive":  "",
		"blkio.io_wait_time_recursive":     "",
		"blkio.io_merged_recursive":        "",
		"blkio.time_recursive":             "",
		"blkio.throttle.io_service_bytes":  throttleServiceBytes,
		"blkio.throttle.io_serviced":       throttleServiced,
	})

	blkio := &BlkioGroup{}
	actualStats := *cgroups.NewStats()
	err := blkio.GetStats(helper.CgroupPath, &actualStats)
	if err != nil {
		t.Fatal(err)
	}

	// Verify expected stats.
	expectedStats := cgroups.BlkioStats{}

	appendBlkioStatEntry(&expectedStats.IoServiceBytesRecursive, 8, 0, 11030528, "Read")
	appendBlkioStatEntry(&expectedStats.IoServiceBytesRecursive, 8, 0, 23, "Write")
	appendBlkioStatEntry(&expectedStats.IoServiceBytesRecursive, 8, 0, 42, "Sync")
	appendBlkioStatEntry(&expectedStats.IoServiceBytesRecursive, 8, 0, 11030528, "Async")
	appendBlkioStatEntry(&expectedStats.IoServiceBytesRecursive, 8, 0, 11030528, "Total")
	appendBlkioStatEntry(&expectedStats.IoServiceBytesRecursive, 252, 0, 11030528, "Read")
	appendBlkioStatEntry(&expectedStats.IoServiceBytesRecursive, 252, 0, 23, "Write")
	appendBlkioStatEntry(&expectedStats.IoServiceBytesRecursive, 252, 0, 42, "Sync")
	appendBlkioStatEntry(&expectedStats.IoServiceBytesRecursive, 252, 0, 11030528, "Async")
	appendBlkioStatEntry(&expectedStats.IoServiceBytesRecursive, 252, 0, 11030528, "Total")

	appendBlkioStatEntry(&expectedStats.IoServicedRecursive, 8, 0, 164, "Read")
	appendBlkioStatEntry(&expectedStats.IoServicedRecursive, 8, 0, 23, "Write")
	appendBlkioStatEntry(&expectedStats.IoServicedRecursive, 8, 0, 42, "Sync")
	appendBlkioStatEntry(&expectedStats.IoServicedRecursive, 8, 0, 164, "Async")
	appendBlkioStatEntry(&expectedStats.IoServicedRecursive, 8, 0, 164, "Total")
	appendBlkioStatEntry(&expectedStats.IoServicedRecursive, 252, 0, 164, "Read")
	appendBlkioStatEntry(&expectedStats.IoServicedRecursive, 252, 0, 23, "Write")
	appendBlkioStatEntry(&expectedStats.IoServicedRecursive, 252, 0, 42, "Sync")
	appendBlkioStatEntry(&expectedStats.IoServicedRecursive, 252, 0, 164, "Async")
	appendBlkioStatEntry(&expectedStats.IoServicedRecursive, 252, 0, 164, "Total")

	expectBlkioStatsEquals(t, expectedStats, actualStats.BlkioStats)
}
