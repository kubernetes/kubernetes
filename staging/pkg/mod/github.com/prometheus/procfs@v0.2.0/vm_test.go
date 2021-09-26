// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build !windows

package procfs

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func newPInt64(i int64) *int64 {
	return &i
}

func TestVM(t *testing.T) {
	fs, err := NewFS(procTestFixtures)
	if err != nil {
		t.Fatal(err)
	}
	got, err := fs.VM()
	if err != nil {
		t.Fatal(err)
	}
	zeroPointer := newPInt64(0)
	lowmemreserveratio := []*int64{newPInt64(256), newPInt64(256), newPInt64(32), zeroPointer, zeroPointer}
	want := &VM{
		AdminReserveKbytes:        newPInt64(8192),
		BlockDump:                 zeroPointer,
		CompactUnevictableAllowed: newPInt64(1),
		DirtyBackgroundBytes:      zeroPointer,
		DirtyBackgroundRatio:      newPInt64(10),
		DirtyBytes:                zeroPointer,
		DirtyExpireCentisecs:      newPInt64(3000),
		DirtyRatio:                newPInt64(20),
		DirtytimeExpireSeconds:    newPInt64(43200),
		DirtyWritebackCentisecs:   newPInt64(500),
		DropCaches:                zeroPointer,
		ExtfragThreshold:          newPInt64(500),
		HugetlbShmGroup:           zeroPointer,
		LaptopMode:                newPInt64(5),
		LegacyVaLayout:            zeroPointer,
		LowmemReserveRatio:        lowmemreserveratio,
		MaxMapCount:               newPInt64(65530),
		MemoryFailureEarlyKill:    zeroPointer,
		MemoryFailureRecovery:     newPInt64(1),
		MinFreeKbytes:             newPInt64(67584),
		MinSlabRatio:              newPInt64(5),
		MinUnmappedRatio:          newPInt64(1),
		MmapMinAddr:               newPInt64(65536),
		NumaStat:                  newPInt64(1),
		NumaZonelistOrder:         "Node",
		NrHugepages:               zeroPointer,
		NrHugepagesMempolicy:      zeroPointer,
		NrOvercommitHugepages:     zeroPointer,
		OomDumpTasks:              newPInt64(1),
		OomKillAllocatingTask:     zeroPointer,
		OvercommitKbytes:          zeroPointer,
		OvercommitMemory:          zeroPointer,
		OvercommitRatio:           newPInt64(50),
		PageCluster:               newPInt64(3),
		PanicOnOom:                zeroPointer,
		PercpuPagelistFraction:    zeroPointer,
		StatInterval:              newPInt64(1),
		Swappiness:                newPInt64(60),
		UserReserveKbytes:         newPInt64(131072),
		VfsCachePressure:          newPInt64(100),
		WatermarkBoostFactor:      newPInt64(15000),
		WatermarkScaleFactor:      newPInt64(10),
		ZoneReclaimMode:           zeroPointer,
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("unexpected power supply class (-want +got):\n%s", diff)
	}
}
