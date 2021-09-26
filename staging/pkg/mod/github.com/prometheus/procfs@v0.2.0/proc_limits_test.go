// Copyright 2018 The Prometheus Authors
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

import "testing"

func TestLimits(t *testing.T) {
	p, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}

	l, err := p.Limits()
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		name string
		want int64
		have int64
	}{
		{name: "cpu time", want: -1, have: l.CPUTime},
		{name: "open files", want: 2048, have: l.OpenFiles},
		{name: "msgqueue size", want: 819200, have: l.MsqqueueSize},
		{name: "nice priority", want: 0, have: l.NicePriority},
		{name: "address space", want: 8589934592, have: l.AddressSpace},
	} {
		if test.want != test.have {
			t.Errorf("want %s %d, have %d", test.name, test.want, test.have)
		}
	}
}
