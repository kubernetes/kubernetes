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

import (
	"testing"
)

func TestProcStatus(t *testing.T) {
	p, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}

	s, err := p.NewStatus()
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		name string
		want int
		have int
	}{
		{name: "Pid", want: 26231, have: s.PID},
		{name: "Tgid", want: 26231, have: s.TGID},
		{name: "VmPeak", want: 58472 * 1024, have: int(s.VmPeak)},
		{name: "VmSize", want: 58440 * 1024, have: int(s.VmSize)},
		{name: "VmLck", want: 0 * 1024, have: int(s.VmLck)},
		{name: "VmPin", want: 0 * 1024, have: int(s.VmPin)},
		{name: "VmHWM", want: 8028 * 1024, have: int(s.VmHWM)},
		{name: "VmRSS", want: 6716 * 1024, have: int(s.VmRSS)},
		{name: "RssAnon", want: 2092 * 1024, have: int(s.RssAnon)},
		{name: "RssFile", want: 4624 * 1024, have: int(s.RssFile)},
		{name: "RssShmem", want: 0 * 1024, have: int(s.RssShmem)},
		{name: "VmData", want: 2580 * 1024, have: int(s.VmData)},
		{name: "VmStk", want: 136 * 1024, have: int(s.VmStk)},
		{name: "VmExe", want: 948 * 1024, have: int(s.VmExe)},
		{name: "VmLib", want: 6816 * 1024, have: int(s.VmLib)},
		{name: "VmPTE", want: 128 * 1024, have: int(s.VmPTE)},
		{name: "VmPMD", want: 12 * 1024, have: int(s.VmPMD)},
		{name: "VmSwap", want: 660 * 1024, have: int(s.VmSwap)},
		{name: "HugetlbPages", want: 0 * 1024, have: int(s.HugetlbPages)},
		{name: "VoluntaryCtxtSwitches", want: 4742839, have: int(s.VoluntaryCtxtSwitches)},
		{name: "NonVoluntaryCtxtSwitches", want: 1727500, have: int(s.NonVoluntaryCtxtSwitches)},
		{name: "TotalCtxtSwitches", want: 4742839 + 1727500, have: int(s.TotalCtxtSwitches())},
	} {
		if test.want != test.have {
			t.Errorf("want %s %d, have %d", test.name, test.want, test.have)
		}
	}
}

func TestProcStatusName(t *testing.T) {
	p, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}
	s, err := p.NewStatus()
	if err != nil {
		t.Fatal(err)
	}
	if want, have := "prometheus", s.Name; want != have {
		t.Errorf("want name %s, have %s", want, have)
	}
}

func TestProcStatusUIDs(t *testing.T) {
	p, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}

	s, err := p.NewStatus()
	if err != nil {
		t.Fatal(err)
	}

	if want, have := [4]string{"1000", "1000", "1000", "0"}, s.UIDs; want != have {
		t.Errorf("want uids %s, have %s", want, have)
	}
}

func TestProcStatusGIDs(t *testing.T) {
	p, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}

	s, err := p.NewStatus()
	if err != nil {
		t.Fatal(err)
	}

	if want, have := [4]string{"1001", "1001", "1001", "0"}, s.GIDs; want != have {
		t.Errorf("want uids %s, have %s", want, have)
	}
}
