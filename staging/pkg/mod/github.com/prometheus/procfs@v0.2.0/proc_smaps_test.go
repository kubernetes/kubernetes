// Copyright 2020 The Prometheus Authors
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
)

func BenchmarkProcSMapsRollup(b *testing.B) {
	fs, err := NewFS(procTestFixtures)
	if err != nil {
		b.Fatalf("Creating pseudo fs from getProcFixtures failed at fixtures/proc with error: %s", err)
	}

	p, err := fs.Proc(26231)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		_, _ = p.ProcSMapsRollup()
	}
}

func TestProcSmapsRollup(t *testing.T) {
	p, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}

	s1, err := p.ProcSMapsRollup()
	if err != nil {
		t.Fatal(err)
	}

	s2, err := p.procSMapsRollupManual()
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name  string
		smaps ProcSMapsRollup
	}{
		{
			name:  "ProcSMapsRollup",
			smaps: s1,
		},
		{
			name:  "procSMapsRollupManual",
			smaps: s2,
		},
	}

	for _, c := range cases {
		for _, test := range []struct {
			name string
			want uint64
			have uint64
		}{
			{name: "Rss", want: 29948 * 1024, have: c.smaps.Rss},
			{name: "Pss", want: 29944 * 1024, have: c.smaps.Pss},
			{name: "SharedClean", want: 4 * 1024, have: c.smaps.SharedClean},
			{name: "SharedDirty", want: 0 * 1024, have: c.smaps.SharedDirty},
			{name: "PrivateClean", want: 15548 * 1024, have: c.smaps.PrivateClean},
			{name: "PrivateDirty", want: 14396 * 1024, have: c.smaps.PrivateDirty},
			{name: "Referenced", want: 24752 * 1024, have: c.smaps.Referenced},
			{name: "Anonymous", want: 20756 * 1024, have: c.smaps.Anonymous},
			{name: "Swap", want: 1940 * 1024, have: c.smaps.Swap},
			{name: "SwapPss", want: 1940 * 1024, have: c.smaps.SwapPss},
		} {
			if test.want != test.have {
				t.Errorf("want %s %s %d, have %d", c.name, test.name, test.want, test.have)
			}
		}
	}
}
