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
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestNetSoftnet(t *testing.T) {
	fs, err := NewFS(procTestFixtures)
	if err != nil {
		t.Fatal(err)
	}

	want := []SoftnetStat{{
		Processed:    0x00015c73,
		Dropped:      0x00020e76,
		TimeSqueezed: 0xf0000769,
	},
		{
			Processed:    0x01663fb2,
			TimeSqueezed: 0x0109a4,
		}}

	got, err := fs.NetSoftnetStat()
	if err != nil {
		t.Fatal(err)
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("unexpected softnet stats(-want +got):\n%s", diff)
	}
}

func TestBadSoftnet(t *testing.T) {
	softNetProcFile = "net/softnet_stat.broken"
	fs, err := NewFS(procTestFixtures)
	if err != nil {
		t.Fatal(err)
	}

	_, err = fs.NetSoftnetStat()
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}
