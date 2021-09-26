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

func TestNetDevParseLine(t *testing.T) {
	const rawLine = `  eth0: 1 2 3    4    5     6          7         8 9  10    11    12    13     14       15          16`

	have, err := NetDev{}.parseLine(rawLine)
	if err != nil {
		t.Fatal(err)
	}

	want := NetDevLine{"eth0", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	if want != *have {
		t.Errorf("want %v, have %v", want, have)
	}
}

func TestNetDev(t *testing.T) {
	fs, err := NewFS(procTestFixtures)
	if err != nil {
		t.Fatal(err)
	}

	netDev, err := fs.NetDev()
	if err != nil {
		t.Fatal(err)
	}

	lines := map[string]NetDevLine{
		"vethf345468": {Name: "vethf345468", RxBytes: 648, RxPackets: 8, TxBytes: 438, TxPackets: 5},
		"lo":          {Name: "lo", RxBytes: 1664039048, RxPackets: 1566805, TxBytes: 1664039048, TxPackets: 1566805},
		"docker0":     {Name: "docker0", RxBytes: 2568, RxPackets: 38, TxBytes: 438, TxPackets: 5},
		"eth0":        {Name: "eth0", RxBytes: 874354587, RxPackets: 1036395, TxBytes: 563352563, TxPackets: 732147},
	}

	if want, have := len(lines), len(netDev); want != have {
		t.Errorf("want %d parsed net/dev lines, have %d", want, have)
	}
	for _, line := range netDev {
		if want, have := lines[line.Name], line; want != have {
			t.Errorf("%s: want %v, have %v", line.Name, want, have)
		}
	}
}

func TestProcNetDev(t *testing.T) {
	p, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}

	netDev, err := p.NetDev()
	if err != nil {
		t.Fatal(err)
	}

	lines := map[string]NetDevLine{
		"lo":   {Name: "lo"},
		"eth0": {Name: "eth0", RxBytes: 438, RxPackets: 5, TxBytes: 648, TxPackets: 8},
	}

	if want, have := len(lines), len(netDev); want != have {
		t.Errorf("want %d parsed net/dev lines, have %d", want, have)
	}
	for _, line := range netDev {
		if want, have := lines[line.Name], line; want != have {
			t.Errorf("%s: want %v, have %v", line.Name, want, have)
		}
	}
}
