//go:build linux
// +build linux

/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package bandwidth

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

var tcClassOutput = `class htb 1:1 root prio 0 rate 10000bit ceil 10000bit burst 1600b cburst 1600b 
class htb 1:2 root prio 0 rate 10000bit ceil 10000bit burst 1600b cburst 1600b 
class htb 1:3 root prio 0 rate 10000bit ceil 10000bit burst 1600b cburst 1600b 
class htb 1:4 root prio 0 rate 10000bit ceil 10000bit burst 1600b cburst 1600b 
`

var tcClassOutput2 = `class htb 1:1 root prio 0 rate 10000bit ceil 10000bit burst 1600b cburst 1600b 
class htb 1:2 root prio 0 rate 10000bit ceil 10000bit burst 1600b cburst 1600b 
class htb 1:3 root prio 0 rate 10000bit ceil 10000bit burst 1600b cburst 1600b 
class htb 1:4 root prio 0 rate 10000bit ceil 10000bit burst 1600b cburst 1600b 
class htb 1:5 root prio 0 rate 10000bit ceil 10000bit burst 1600b cburst 1600b 
`

func TestNextClassID(t *testing.T) {
	tests := []struct {
		output    string
		expectErr bool
		expected  int
		err       error
	}{
		{
			output:   tcClassOutput,
			expected: 5,
		},
		{
			output:   "\n",
			expected: 1,
		},
		{
			expected:  -1,
			expectErr: true,
			err:       errors.New("test error"),
		},
	}
	for _, test := range tests {
		fcmd := fakeexec.FakeCmd{
			CombinedOutputScript: []fakeexec.FakeAction{
				func() ([]byte, []byte, error) { return []byte(test.output), nil, test.err },
			},
		}
		fexec := fakeexec.FakeExec{
			CommandScript: []fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fcmd, cmd, args...)
				},
			},
		}
		shaper := &tcShaper{e: &fexec}
		class, err := shaper.nextClassID()
		if test.expectErr {
			if err == nil {
				t.Errorf("unexpected non-error")
			}
		} else {
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if class != test.expected {
				t.Errorf("expected: %d, found %d", test.expected, class)
			}
		}
	}
}

func TestHexCIDR(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		output    string
		expectErr bool
	}{
		{
			name:   "IPv4 masked",
			input:  "1.2.3.4/16",
			output: "01020000/ffff0000",
		},
		{
			name:   "IPv4 host",
			input:  "172.17.0.2/32",
			output: "ac110002/ffffffff",
		},
		{
			name:   "IPv6 masked",
			input:  "2001:dead:beef::cafe/64",
			output: "2001deadbeef00000000000000000000/ffffffffffffffff0000000000000000",
		},
		{
			name:   "IPv6 host",
			input:  "2001::5/128",
			output: "20010000000000000000000000000005/ffffffffffffffffffffffffffffffff",
		},
		{
			name:      "invalid CIDR",
			input:     "foo",
			expectErr: true,
		},
	}
	for _, test := range tests {
		output, err := hexCIDR(test.input)
		if test.expectErr {
			if err == nil {
				t.Errorf("case %s: unexpected non-error", test.name)
			}
		} else {
			if err != nil {
				t.Errorf("case %s: unexpected error: %v", test.name, err)
			}
			if output != test.output {
				t.Errorf("case %s: expected: %s, saw: %s",
					test.name, test.output, output)
			}
		}
	}
}

func TestAsciiCIDR(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		output    string
		expectErr bool
	}{
		{
			name:   "IPv4",
			input:  "01020000/ffff0000",
			output: "1.2.0.0/16",
		},
		{
			name:   "IPv4 host",
			input:  "ac110002/ffffffff",
			output: "172.17.0.2/32",
		},
		{
			name:   "IPv6",
			input:  "2001deadbeef00000000000000000000/ffffffffffffffff0000000000000000",
			output: "2001:dead:beef::/64",
		},
		{
			name:   "IPv6 host",
			input:  "20010000000000000000000000000005/ffffffffffffffffffffffffffffffff",
			output: "2001::5/128",
		},
		{
			name:      "invalid CIDR",
			input:     "malformed",
			expectErr: true,
		},
		{
			name:      "non-hex IP",
			input:     "nonhex/32",
			expectErr: true,
		},
		{
			name:      "non-hex mask",
			input:     "01020000/badmask",
			expectErr: true,
		},
	}
	for _, test := range tests {
		output, err := asciiCIDR(test.input)
		if test.expectErr {
			if err == nil {
				t.Errorf("case %s: unexpected non-error", test.name)
			}
		} else {
			if err != nil {
				t.Errorf("case %s: unexpected error: %v", test.name, err)
			}
			if output != test.output {
				t.Errorf("case %s: expected: %s, saw: %s",
					test.name, test.output, output)
			}
		}
	}
}

var tcFilterOutput = `filter parent 1: protocol ip pref 1 u32 
filter parent 1: protocol ip pref 1 u32 fh 800: ht divisor 1 
filter parent 1: protocol ip pref 1 u32 fh 800::800 order 2048 key ht 800 bkt 0 flowid 1:1 
  match ac110002/ffffffff at 16
filter parent 1: protocol ip pref 1 u32 fh 800::801 order 2049 key ht 800 bkt 0 flowid 1:2 
  match 01020000/ffff0000 at 16
`
var tcFilterOutputNewVersion = `filter parent 1: protocol ip pref 1 u32
filter parent 1: protocol ip pref 1 u32 chain 0 fh 800: ht divisor 1
filter parent 1: protocol ip pref 1 u32 chain 0 fh 800::800 order 2048 key ht 800 bkt 0 flowid 1:1 not_in_hw
  match ac110002/ffffffff at 16
filter parent 1: protocol ip pref 1 u32 chain 0 fh 800::801 order 2049 key ht 800 bkt 0 flowid 1:2 not_in_hw
  match 01020000/ffff0000 at 16
`

func TestFindCIDRClass(t *testing.T) {
	tests := []struct {
		cidr           string
		output         string
		expectErr      bool
		expectNotFound bool
		expectedClass  string
		expectedHandle string
		err            error
	}{
		{
			cidr:           "172.17.0.2/32",
			output:         tcFilterOutput,
			expectedClass:  "1:1",
			expectedHandle: "800::800",
		},
		{
			cidr:           "1.2.3.4/16",
			output:         tcFilterOutput,
			expectedClass:  "1:2",
			expectedHandle: "800::801",
		},
		{
			cidr:           "2.2.3.4/16",
			output:         tcFilterOutput,
			expectNotFound: true,
		},
		{
			cidr:           "172.17.0.2/32",
			output:         tcFilterOutputNewVersion,
			expectedClass:  "1:1",
			expectedHandle: "800::800",
		},
		{
			cidr:           "1.2.3.4/16",
			output:         tcFilterOutputNewVersion,
			expectedClass:  "1:2",
			expectedHandle: "800::801",
		},
		{
			cidr:           "2.2.3.4/16",
			output:         tcFilterOutputNewVersion,
			expectNotFound: true,
		},
		{
			err:       errors.New("test error"),
			expectErr: true,
		},
	}
	for _, test := range tests {
		fcmd := fakeexec.FakeCmd{
			CombinedOutputScript: []fakeexec.FakeAction{
				func() ([]byte, []byte, error) { return []byte(test.output), nil, test.err },
			},
		}
		fexec := fakeexec.FakeExec{
			CommandScript: []fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd {
					return fakeexec.InitFakeCmd(&fcmd, cmd, args...)
				},
			},
		}
		shaper := &tcShaper{e: &fexec}
		classAndHandle, found, err := shaper.findCIDRClass(test.cidr)
		if test.expectErr {
			if err == nil {
				t.Errorf("unexpected non-error")
			}
		} else {
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if test.expectNotFound {
				if found {
					t.Errorf("unexpectedly found an interface: %s", classAndHandle)
				}
			} else {
				if classAndHandle[0][0] != test.expectedClass {
					t.Errorf("expected class: %s, found %s", test.expectedClass, classAndHandle)
				}
				if classAndHandle[0][1] != test.expectedHandle {
					t.Errorf("expected handle: %s, found %s", test.expectedHandle, classAndHandle)
				}
			}
		}
	}
}

func TestGetCIDRs(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return []byte(tcFilterOutput), nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd {
				return fakeexec.InitFakeCmd(&fcmd, cmd, args...)
			},
		},
	}
	shaper := &tcShaper{e: &fexec}
	cidrs, err := shaper.GetCIDRs()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	expectedCidrs := []string{"172.17.0.2/32", "1.2.0.0/16"}
	if !reflect.DeepEqual(cidrs, expectedCidrs) {
		t.Errorf("expected: %v, saw: %v", expectedCidrs, cidrs)
	}
}

func TestLimit(t *testing.T) {
	tests := []struct {
		cidr          string
		ingress       *resource.Quantity
		egress        *resource.Quantity
		expectErr     bool
		expectedCalls int
		err           error
	}{
		{
			cidr:          "1.2.3.4/32",
			ingress:       resource.NewQuantity(10, resource.DecimalSI),
			egress:        resource.NewQuantity(20, resource.DecimalSI),
			expectedCalls: 6,
		},
		{
			cidr:          "1.2.3.4/32",
			ingress:       resource.NewQuantity(10, resource.DecimalSI),
			egress:        nil,
			expectedCalls: 3,
		},
		{
			cidr:          "1.2.3.4/32",
			ingress:       nil,
			egress:        resource.NewQuantity(20, resource.DecimalSI),
			expectedCalls: 3,
		},
		{
			cidr:          "1.2.3.4/32",
			ingress:       nil,
			egress:        nil,
			expectedCalls: 0,
		},
		{
			err:       errors.New("test error"),
			ingress:   resource.NewQuantity(10, resource.DecimalSI),
			egress:    resource.NewQuantity(20, resource.DecimalSI),
			expectErr: true,
		},
	}

	for _, test := range tests {
		fcmd := fakeexec.FakeCmd{
			CombinedOutputScript: []fakeexec.FakeAction{
				func() ([]byte, []byte, error) { return []byte(tcClassOutput), nil, test.err },
				func() ([]byte, []byte, error) { return []byte{}, nil, test.err },
				func() ([]byte, []byte, error) { return []byte{}, nil, test.err },
				func() ([]byte, []byte, error) { return []byte(tcClassOutput2), nil, test.err },
				func() ([]byte, []byte, error) { return []byte{}, nil, test.err },
				func() ([]byte, []byte, error) { return []byte{}, nil, test.err },
			},
		}

		fexec := fakeexec.FakeExec{
			CommandScript: []fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			},
		}
		iface := "cbr0"
		shaper := &tcShaper{e: &fexec, iface: iface}
		if err := shaper.Limit(test.cidr, test.ingress, test.egress); err != nil && !test.expectErr {
			t.Errorf("unexpected error: %v", err)
			return
		} else if err == nil && test.expectErr {
			t.Error("unexpected non-error")
			return
		}
		// No more testing in the error case
		if test.expectErr {
			if fcmd.CombinedOutputCalls != 1 {
				t.Errorf("unexpected number of calls: %d, expected: 1", fcmd.CombinedOutputCalls)
			}
			return
		}

		if fcmd.CombinedOutputCalls != test.expectedCalls {
			t.Errorf("unexpected number of calls: %d, expected: %d", fcmd.CombinedOutputCalls, test.expectedCalls)
		}

		for ix := range fcmd.CombinedOutputLog {
			output := fcmd.CombinedOutputLog[ix]
			if output[0] != "tc" {
				t.Errorf("unexpected command: %s, expected tc", output[0])
			}
			if output[4] != iface {
				t.Errorf("unexpected interface: %s, expected %s (%v)", output[4], iface, output)
			}
			if ix == 1 {
				var expectedRate string
				if test.ingress != nil {
					expectedRate = makeKBitString(test.ingress)
				} else {
					expectedRate = makeKBitString(test.egress)
				}
				if output[11] != expectedRate {
					t.Errorf("unexpected ingress: %s, expected: %s", output[11], expectedRate)
				}
				if output[8] != "1:5" {
					t.Errorf("unexpected class: %s, expected: %s", output[8], "1:5")
				}
			}
			if ix == 2 {
				if output[15] != test.cidr {
					t.Errorf("unexpected cidr: %s, expected: %s", output[15], test.cidr)
				}
				if output[17] != "1:5" {
					t.Errorf("unexpected class: %s, expected: %s", output[17], "1:5")
				}
			}
			if ix == 4 {
				if output[11] != makeKBitString(test.egress) {
					t.Errorf("unexpected egress: %s, expected: %s", output[11], makeKBitString(test.egress))
				}
				if output[8] != "1:6" {
					t.Errorf("unexpected class: %s, expected: %s", output[8], "1:6")
				}
			}
			if ix == 5 {
				if output[15] != test.cidr {
					t.Errorf("unexpected cidr: %s, expected: %s", output[15], test.cidr)
				}
				if output[17] != "1:6" {
					t.Errorf("unexpected class: %s, expected: %s", output[17], "1:5")
				}
			}
		}
	}
}

func TestReset(t *testing.T) {
	tests := []struct {
		cidr           string
		err            error
		expectErr      bool
		expectedHandle string
		expectedClass  string
	}{
		{
			cidr:           "1.2.3.4/16",
			expectedHandle: "800::801",
			expectedClass:  "1:2",
		},
		{
			cidr:           "172.17.0.2/32",
			expectedHandle: "800::800",
			expectedClass:  "1:1",
		},
		{
			err:       errors.New("test error"),
			expectErr: true,
		},
	}
	for _, test := range tests {
		fcmd := fakeexec.FakeCmd{
			CombinedOutputScript: []fakeexec.FakeAction{
				func() ([]byte, []byte, error) { return []byte(tcFilterOutput), nil, test.err },
				func() ([]byte, []byte, error) { return []byte{}, nil, test.err },
				func() ([]byte, []byte, error) { return []byte{}, nil, test.err },
			},
		}

		fexec := fakeexec.FakeExec{
			CommandScript: []fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			},
		}
		iface := "cbr0"
		shaper := &tcShaper{e: &fexec, iface: iface}

		if err := shaper.Reset(test.cidr); err != nil && !test.expectErr {
			t.Errorf("unexpected error: %v", err)
			return
		} else if test.expectErr && err == nil {
			t.Error("unexpected non-error")
			return
		}

		// No more testing in the error case
		if test.expectErr {
			if fcmd.CombinedOutputCalls != 1 {
				t.Errorf("unexpected number of calls: %d, expected: 1", fcmd.CombinedOutputCalls)
			}
			return
		}

		if fcmd.CombinedOutputCalls != 3 {
			t.Errorf("unexpected number of calls: %d, expected: 3", fcmd.CombinedOutputCalls)
		}

		for ix := range fcmd.CombinedOutputLog {
			output := fcmd.CombinedOutputLog[ix]
			if output[0] != "tc" {
				t.Errorf("unexpected command: %s, expected tc", output[0])
			}
			if output[4] != iface {
				t.Errorf("unexpected interface: %s, expected %s (%v)", output[4], iface, output)
			}
			if ix == 1 && output[12] != test.expectedHandle {
				t.Errorf("unexpected handle: %s, expected: %s", output[12], test.expectedHandle)
			}
			if ix == 2 && output[8] != test.expectedClass {
				t.Errorf("unexpected class: %s, expected: %s", output[8], test.expectedClass)
			}
		}
	}
}

var tcQdisc = "qdisc htb 1: root refcnt 2 r2q 10 default 30 direct_packets_stat 0\n"

func TestReconcileInterfaceExists(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return []byte(tcQdisc), nil, nil },
		},
	}

	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	iface := "cbr0"
	shaper := &tcShaper{e: &fexec, iface: iface}
	err := shaper.ReconcileInterface()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("unexpected number of calls: %d", fcmd.CombinedOutputCalls)
	}

	output := fcmd.CombinedOutputLog[0]
	if len(output) != 5 {
		t.Errorf("unexpected command: %v", output)
	}
	if output[0] != "tc" {
		t.Errorf("unexpected command: %s", output[0])
	}
	if output[4] != iface {
		t.Errorf("unexpected interface: %s, expected %s", output[4], iface)
	}
	if output[2] != "show" {
		t.Errorf("unexpected action: %s", output[2])
	}
}

func testReconcileInterfaceHasNoData(t *testing.T, output string) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return []byte(output), nil, nil },
			func() ([]byte, []byte, error) { return []byte(output), nil, nil },
		},
	}

	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	iface := "cbr0"
	shaper := &tcShaper{e: &fexec, iface: iface}
	err := shaper.ReconcileInterface()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("unexpected number of calls: %d", fcmd.CombinedOutputCalls)
	}

	for ix, output := range fcmd.CombinedOutputLog {
		if output[0] != "tc" {
			t.Errorf("unexpected command: %s", output[0])
		}
		if output[4] != iface {
			t.Errorf("unexpected interface: %s, expected %s", output[4], iface)
		}
		if ix == 0 {
			if len(output) != 5 {
				t.Errorf("unexpected command: %v", output)
			}
			if output[2] != "show" {
				t.Errorf("unexpected action: %s", output[2])
			}
		}
		if ix == 1 {
			if len(output) != 11 {
				t.Errorf("unexpected command: %v", output)
			}
			if output[2] != "add" {
				t.Errorf("unexpected action: %s", output[2])
			}
			if output[7] != "1:" {
				t.Errorf("unexpected root class: %s", output[7])
			}
			if output[8] != "htb" {
				t.Errorf("unexpected qdisc algo: %s", output[8])
			}
		}
	}
}

func TestReconcileInterfaceDoesntExist(t *testing.T) {
	testReconcileInterfaceHasNoData(t, "\n")
}

var tcQdiscNoqueue = "qdisc noqueue 0: root refcnt 2 \n"

func TestReconcileInterfaceExistsWithNoqueue(t *testing.T) {
	testReconcileInterfaceHasNoData(t, tcQdiscNoqueue)
}

var tcQdiscWrong = []string{
	"qdisc htb 2: root refcnt 2 r2q 10 default 30 direct_packets_stat 0\n",
	"qdisc foo 1: root refcnt 2 r2q 10 default 30 direct_packets_stat 0\n",
}

func TestReconcileInterfaceIsWrong(t *testing.T) {
	for _, test := range tcQdiscWrong {
		fcmd := fakeexec.FakeCmd{
			CombinedOutputScript: []fakeexec.FakeAction{
				func() ([]byte, []byte, error) { return []byte(test), nil, nil },
				func() ([]byte, []byte, error) { return []byte("\n"), nil, nil },
				func() ([]byte, []byte, error) { return []byte("\n"), nil, nil },
			},
		}

		fexec := fakeexec.FakeExec{
			CommandScript: []fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			},
		}
		iface := "cbr0"
		shaper := &tcShaper{e: &fexec, iface: iface}
		err := shaper.ReconcileInterface()
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		if fcmd.CombinedOutputCalls != 3 {
			t.Errorf("unexpected number of calls: %d", fcmd.CombinedOutputCalls)
		}

		for ix, output := range fcmd.CombinedOutputLog {
			if output[0] != "tc" {
				t.Errorf("unexpected command: %s", output[0])
			}
			if output[4] != iface {
				t.Errorf("unexpected interface: %s, expected %s", output[4], iface)
			}
			if ix == 0 {
				if len(output) != 5 {
					t.Errorf("unexpected command: %v", output)
				}
				if output[2] != "show" {
					t.Errorf("unexpected action: %s", output[2])
				}
			}
			if ix == 1 {
				if len(output) != 8 {
					t.Errorf("unexpected command: %v", output)
				}
				if output[2] != "delete" {
					t.Errorf("unexpected action: %s", output[2])
				}
				if output[7] != strings.Split(test, " ")[2] {
					t.Errorf("unexpected class: %s, expected: %s", output[7], strings.Split(test, " ")[2])
				}
			}
			if ix == 2 {
				if len(output) != 11 {
					t.Errorf("unexpected command: %v", output)
				}
				if output[7] != "1:" {
					t.Errorf("unexpected root class: %s", output[7])
				}
				if output[8] != "htb" {
					t.Errorf("unexpected qdisc algo: %s", output[8])
				}
			}
		}
	}
}
