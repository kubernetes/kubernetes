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

func TestProcIO(t *testing.T) {
	p, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}

	s, err := p.IO()
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		name string
		want int64
		have int64
	}{
		{name: "RChar", want: 750339, have: int64(s.RChar)},
		{name: "WChar", want: 818609, have: int64(s.WChar)},
		{name: "SyscR", want: 7405, have: int64(s.SyscR)},
		{name: "SyscW", want: 5245, have: int64(s.SyscW)},
		{name: "ReadBytes", want: 1024, have: int64(s.ReadBytes)},
		{name: "WriteBytes", want: 2048, have: int64(s.WriteBytes)},
		{name: "CancelledWriteBytes", want: -1024, have: s.CancelledWriteBytes},
	} {
		if test.want != test.have {
			t.Errorf("want %s %d, have %d", test.name, test.want, test.have)
		}
	}
}
