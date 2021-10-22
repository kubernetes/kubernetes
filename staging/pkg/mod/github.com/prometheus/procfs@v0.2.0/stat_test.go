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

func TestStat(t *testing.T) {
	s, err := getProcFixtures(t).Stat()
	if err != nil {
		t.Fatal(err)
	}

	// cpu
	if want, have := float64(301854)/userHZ, s.CPUTotal.User; want != have {
		t.Errorf("want cpu/user %v, have %v", want, have)
	}
	if want, have := float64(31)/userHZ, s.CPU[7].SoftIRQ; want != have {
		t.Errorf("want cpu7/softirq %v, have %v", want, have)
	}

	// intr
	if want, have := uint64(8885917), s.IRQTotal; want != have {
		t.Errorf("want irq/total %d, have %d", want, have)
	}
	if want, have := uint64(1), s.IRQ[8]; want != have {
		t.Errorf("want irq8 %d, have %d", want, have)
	}

	// ctxt
	if want, have := uint64(38014093), s.ContextSwitches; want != have {
		t.Errorf("want context switches (ctxt) %d, have %d", want, have)
	}

	// btime
	if want, have := uint64(1418183276), s.BootTime; want != have {
		t.Errorf("want boot time (btime) %d, have %d", want, have)
	}

	// processes
	if want, have := uint64(26442), s.ProcessCreated; want != have {
		t.Errorf("want process created (processes) %d, have %d", want, have)
	}

	// procs_running
	if want, have := uint64(2), s.ProcessesRunning; want != have {
		t.Errorf("want processes running (procs_running) %d, have %d", want, have)
	}

	// procs_blocked
	if want, have := uint64(1), s.ProcessesBlocked; want != have {
		t.Errorf("want processes blocked (procs_blocked) %d, have %d", want, have)
	}

	// softirq
	if want, have := uint64(5057579), s.SoftIRQTotal; want != have {
		t.Errorf("want softirq total %d, have %d", want, have)
	}

	if want, have := uint64(508444), s.SoftIRQ.Rcu; want != have {
		t.Errorf("want softirq RCU %d, have %d", want, have)
	}

}
