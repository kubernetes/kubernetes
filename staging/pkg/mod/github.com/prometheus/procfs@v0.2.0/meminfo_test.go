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

package procfs

import (
	"reflect"
	"testing"
)

func TestMeminfo(t *testing.T) {
	expected := Meminfo{
		MemTotal:          15666184,
		MemFree:           440324,
		Buffers:           1020128,
		Cached:            12007640,
		SwapCached:        0,
		Active:            6761276,
		Inactive:          6532708,
		ActiveAnon:        267256,
		InactiveAnon:      268,
		ActiveFile:        6494020,
		InactiveFile:      6532440,
		Unevictable:       0,
		Mlocked:           0,
		SwapTotal:         0,
		SwapFree:          0,
		Dirty:             768,
		Writeback:         0,
		AnonPages:         266216,
		Mapped:            44204,
		Shmem:             1308,
		Slab:              1807264,
		SReclaimable:      1738124,
		SUnreclaim:        69140,
		KernelStack:       1616,
		PageTables:        5288,
		NFSUnstable:       0,
		Bounce:            0,
		WritebackTmp:      0,
		CommitLimit:       7833092,
		CommittedAS:       530844,
		VmallocTotal:      34359738367,
		VmallocUsed:       36596,
		VmallocChunk:      34359637840,
		HardwareCorrupted: 0,
		AnonHugePages:     12288,
		HugePagesTotal:    0,
		HugePagesFree:     0,
		HugePagesRsvd:     0,
		HugePagesSurp:     0,
		Hugepagesize:      2048,
		DirectMap4k:       91136,
		DirectMap2M:       16039936,
	}

	have, err := getProcFixtures(t).Meminfo()
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(have, expected) {
		t.Logf("have: %+v", have)
		t.Logf("expected: %+v", expected)
		t.Errorf("structs are not equal")
	}
}
