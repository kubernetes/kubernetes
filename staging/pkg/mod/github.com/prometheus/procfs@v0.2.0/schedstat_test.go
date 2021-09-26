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
)

func TestSchedstat(t *testing.T) {
	stats, err := getProcFixtures(t).Schedstat()
	if err != nil {
		t.Fatal(err)
	}

	if len(stats.CPUs) != 2 {
		t.Errorf("expected 2 CPUs, got %v", len(stats.CPUs))
	}

	var cpu *SchedstatCPU
	for _, cpu = range stats.CPUs {
		if cpu.CPUNum == "0" {
			break
		}
	}

	if cpu == nil || cpu.CPUNum != "0" {
		t.Error("could not find cpu0")
	}

	if want, have := uint64(2045936778163039), cpu.RunningNanoseconds; want != have {
		t.Errorf("want RunningNanoseconds %v, have %v", want, have)
	}

	if want, have := uint64(343796328169361), cpu.WaitingNanoseconds; want != have {
		t.Errorf("want WaitingNanoseconds %v, have %v", want, have)
	}

	if want, have := uint64(4767485306), cpu.RunTimeslices; want != have {
		t.Errorf("want RunTimeslices %v, have %v", want, have)
	}
}

func TestProcSchedstat(t *testing.T) {
	p1, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}

	schedstat, err := p1.Schedstat()
	if err != nil {
		t.Fatal(err)
	}

	if want, have := uint64(411605849), schedstat.RunningNanoseconds; want != have {
		t.Errorf("want RunningNanoseconds %v, have %v", want, have)
	}

	if want, have := uint64(93680043), schedstat.WaitingNanoseconds; want != have {
		t.Errorf("want WaitingNanoseconds %v, have %v", want, have)
	}

	if want, have := uint64(79), schedstat.RunTimeslices; want != have {
		t.Errorf("want RunTimeslices %v, have %v", want, have)
	}
}

func TestProcSchedstatErrors(t *testing.T) {
	p1, err := getProcFixtures(t).Proc(26232)
	if err != nil {
		t.Fatal(err)
	}

	_, err = p1.Schedstat()
	if err == nil {
		t.Error("proc 26232 doesn't have schedstat -- should have gotten an error")
	}

	p2, err := getProcFixtures(t).Proc(26233)
	if err != nil {
		t.Fatal(err)
	}

	_, err = p2.Schedstat()
	if err == nil {
		t.Error("proc 26233 has malformed schedstat -- should have gotten an error")
	}
}

// schedstat can have a 2nd line: it should be ignored
func TestProcSchedstatMultipleLines(t *testing.T) {
	schedstat, err := parseProcSchedstat("123 456 789\n10 11\n")
	if err != nil {
		t.Fatal(err)
	}
	if want, have := uint64(123), schedstat.RunningNanoseconds; want != have {
		t.Errorf("want RunningNanoseconds %v, have %v", want, have)
	}
	if want, have := uint64(456), schedstat.WaitingNanoseconds; want != have {
		t.Errorf("want WaitingNanoseconds %v, have %v", want, have)
	}
	if want, have := uint64(789), schedstat.RunTimeslices; want != have {
		t.Errorf("want RunTimeslices %v, have %v", want, have)
	}
}

func TestProcSchedstatUnparsableInt(t *testing.T) {
	if _, err := parseProcSchedstat("abc 456 789\n"); err == nil {
		t.Error("schedstat should have been unparsable\n")
	}

	if _, err := parseProcSchedstat("123 abc 789\n"); err == nil {
		t.Error("schedstat should have been unparsable\n")
	}

	if _, err := parseProcSchedstat("123 456 abc\n"); err == nil {
		t.Error("schedstat should have been unparsable\n")
	}
}
