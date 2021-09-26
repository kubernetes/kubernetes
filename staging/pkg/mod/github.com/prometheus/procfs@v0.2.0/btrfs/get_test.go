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

package btrfs

import "testing"

type testVector struct {
	uuid, label        string
	devices, features  int
	data, meta, system alloc
}

type alloc struct {
	layout string
	size   uint64
	ratio  float64
}

func TestFSBtrfsStats(t *testing.T) {
	btrfs, err := NewFS("../fixtures/sys")
	if err != nil {
		t.Fatalf("failed to access Btrfs filesystem: %v", err)
	}
	stats, err := btrfs.Stats()
	if err != nil {
		t.Fatalf("failed to parse Btrfs stats: %v", err)
	}

	tests := []testVector{
		{
			uuid:     "0abb23a9-579b-43e6-ad30-227ef47fcb9d",
			label:    "fixture",
			devices:  2,
			features: 4,
			data:     alloc{"raid0", 2147483648, 1},
			meta:     alloc{"raid1", 1073741824, 2},
			system:   alloc{"raid1", 8388608, 2},
		},
		{
			uuid:     "7f07c59f-6136-449c-ab87-e1cf2328731b",
			label:    "",
			devices:  4,
			features: 5,
			data:     alloc{"raid5", 644087808, 4. / 3.},
			meta:     alloc{"raid6", 429391872, 4. / 2.},
			system:   alloc{"raid6", 16777216, 4. / 2.},
		},
	}

	if l := len(stats); l != len(tests) {
		t.Fatalf("unexpected number of btrfs stats: %d", l)
	}

	for i, tt := range tests {
		if want, got := tt.uuid, stats[i].UUID; want != got {
			t.Errorf("fs %q unexpected stats name:\nwant: %q\nhave: %q", tt.uuid, want, got)
		}

		if want, got := tt.devices, len(stats[i].Devices); want != got {
			t.Errorf("fs %q unexpected number of devices:\nwant: %d\nhave: %d", tt.uuid, want, got)
		}

		if want, got := tt.features, len(stats[i].Features); want != got {
			t.Errorf("fs %q unexpected number of features:\nwant: %d\nhave: %d", tt.uuid, want, got)
		}

		if want, got := tt.data.size, stats[i].Allocation.Data.TotalBytes; want != got {
			t.Errorf("fs %q unexpected data size:\nwant: %d\nhave: %d", tt.uuid, want, got)
		}

		if want, got := tt.meta.size, stats[i].Allocation.Metadata.TotalBytes; want != got {
			t.Errorf("fs %q unexpected metadata size:\nwant: %d\nhave: %d", tt.uuid, want, got)
		}

		if want, got := tt.system.size, stats[i].Allocation.System.TotalBytes; want != got {
			t.Errorf("fs %q unexpected system size:\nwant: %d\nhave: %d", tt.uuid, want, got)
		}

		if want, got := tt.data.ratio, stats[i].Allocation.Data.Layouts[tt.data.layout].Ratio; want != got {
			t.Errorf("fs %q unexpected data ratio:\nwant: %f\nhave: %f", tt.uuid, want, got)
		}

		if want, got := tt.meta.ratio, stats[i].Allocation.Metadata.Layouts[tt.meta.layout].Ratio; want != got {
			t.Errorf("fs %q unexpected metadata ratio:\nwant: %f\nhave: %f", tt.uuid, want, got)
		}

		if want, got := tt.system.ratio, stats[i].Allocation.System.Layouts[tt.system.layout].Ratio; want != got {
			t.Errorf("fs %q unexpected system ratio:\nwant: %f\nhave: %f", tt.uuid, want, got)
		}
	}
}
