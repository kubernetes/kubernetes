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

// Package xfs provides access to statistics exposed by the XFS filesystem.
package xfs_test

import (
	"testing"

	"github.com/prometheus/procfs/xfs"
)

func TestReadProcStat(t *testing.T) {
	xfs, err := xfs.NewFS("../fixtures/proc", "../fixtures/sys")
	if err != nil {
		t.Fatalf("failed to access xfs fs: %v", err)
	}
	stats, err := xfs.ProcStat()
	if err != nil {
		t.Fatalf("failed to parse XFS stats: %v", err)
	}

	// Very lightweight test just to sanity check the path used
	// to open XFS stats. Heavier tests in package xfs.
	if want, got := uint32(92447), stats.ExtentAllocation.ExtentsAllocated; want != got {
		t.Errorf("unexpected extents allocated:\nwant: %d\nhave: %d", want, got)
	}
}

func TestReadSysStats(t *testing.T) {
	xfs, err := xfs.NewFS("../fixtures/proc", "../fixtures/sys")
	if err != nil {
		t.Fatalf("failed to access xfs fs: %v", err)
	}
	stats, err := xfs.SysStats()
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
