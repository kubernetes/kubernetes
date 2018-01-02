// Copyright 2017 The Prometheus Authors
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

package sysfs

import "testing"

func TestNewFS(t *testing.T) {
	if _, err := NewFS("foobar"); err == nil {
		t.Error("want NewFS to fail for non-existing mount point")
	}

	if _, err := NewFS("doc.go"); err == nil {
		t.Error("want NewFS to fail if mount point is not a directory")
	}
}

func TestFSXFSStats(t *testing.T) {
	stats, err := FS("fixtures").XFSStats()
	if err != nil {
		t.Fatalf("failed to parse XFS stats: %v", err)
	}

	tests := []struct {
		name      string
		allocated uint32
	}{
		{
			name:      "sda1",
			allocated: 1,
		},
		{
			name:      "sdb1",
			allocated: 2,
		},
	}

	const expect = 2

	if l := len(stats); l != expect {
		t.Fatalf("unexpected number of XFS stats: %d", l)
	}
	if l := len(tests); l != expect {
		t.Fatalf("unexpected number of tests: %d", l)
	}

	for i, tt := range tests {
		if want, got := tt.name, stats[i].Name; want != got {
			t.Errorf("unexpected stats name:\nwant: %q\nhave: %q", want, got)
		}

		if want, got := tt.allocated, stats[i].ExtentAllocation.ExtentsAllocated; want != got {
			t.Errorf("unexpected extents allocated:\nwant: %d\nhave: %d", want, got)
		}
	}
}
