package fs

import (
	"fmt"
	"log"
	"testing"

	"github.com/docker/libcontainer/cgroups"
)

func blkioStatEntryEquals(expected, actual []cgroups.BlkioStatEntry) error {
	if len(expected) != len(actual) {
		return fmt.Errorf("blkioStatEntries length do not match")
	}
	for i, expValue := range expected {
		actValue := actual[i]
		if expValue != actValue {
			return fmt.Errorf("Expected blkio stat entry %v but found %v", expValue, actValue)
		}
	}
	return nil
}

func expectBlkioStatsEquals(t *testing.T, expected, actual cgroups.BlkioStats) {
	if err := blkioStatEntryEquals(expected.IoServiceBytesRecursive, actual.IoServiceBytesRecursive); err != nil {
		log.Printf("blkio IoServiceBytesRecursive do not match - %s\n", err)
		t.Fail()
	}

	if err := blkioStatEntryEquals(expected.IoServicedRecursive, actual.IoServicedRecursive); err != nil {
		log.Printf("blkio IoServicedRecursive do not match - %s\n", err)
		t.Fail()
	}

	if err := blkioStatEntryEquals(expected.IoQueuedRecursive, actual.IoQueuedRecursive); err != nil {
		log.Printf("blkio IoQueuedRecursive do not match - %s\n", err)
		t.Fail()
	}

	if err := blkioStatEntryEquals(expected.SectorsRecursive, actual.SectorsRecursive); err != nil {
		log.Printf("blkio SectorsRecursive do not match - %s\n", err)
		t.Fail()
	}

	if err := blkioStatEntryEquals(expected.IoServiceTimeRecursive, actual.IoServiceTimeRecursive); err != nil {
		log.Printf("blkio IoServiceTimeRecursive do not match - %s\n", err)
		t.Fail()
	}

	if err := blkioStatEntryEquals(expected.IoWaitTimeRecursive, actual.IoWaitTimeRecursive); err != nil {
		log.Printf("blkio IoWaitTimeRecursive do not match - %s\n", err)
		t.Fail()
	}

	if err := blkioStatEntryEquals(expected.IoMergedRecursive, actual.IoMergedRecursive); err != nil {
		log.Printf("blkio IoMergedRecursive do not match - %v vs %v\n", expected.IoMergedRecursive, actual.IoMergedRecursive)
		t.Fail()
	}

	if err := blkioStatEntryEquals(expected.IoTimeRecursive, actual.IoTimeRecursive); err != nil {
		log.Printf("blkio IoTimeRecursive do not match - %s\n", err)
		t.Fail()
	}
}

func expectThrottlingDataEquals(t *testing.T, expected, actual cgroups.ThrottlingData) {
	if expected != actual {
		log.Printf("Expected throttling data %v but found %v\n", expected, actual)
		t.Fail()
	}
}

func expectMemoryStatEquals(t *testing.T, expected, actual cgroups.MemoryStats) {
	if expected.Usage != actual.Usage {
		log.Printf("Expected memory usage %d but found %d\n", expected.Usage, actual.Usage)
		t.Fail()
	}
	if expected.MaxUsage != actual.MaxUsage {
		log.Printf("Expected memory max usage %d but found %d\n", expected.MaxUsage, actual.MaxUsage)
		t.Fail()
	}
	for key, expValue := range expected.Stats {
		actValue, ok := actual.Stats[key]
		if !ok {
			log.Printf("Expected memory stat key %s not found\n", key)
			t.Fail()
		}
		if expValue != actValue {
			log.Printf("Expected memory stat value %d but found %d\n", expValue, actValue)
			t.Fail()
		}
	}
	if expected.Failcnt != actual.Failcnt {
		log.Printf("Expected memory failcnt %d but found %d\n", expected.Failcnt, actual.Failcnt)
		t.Fail()
	}
}
