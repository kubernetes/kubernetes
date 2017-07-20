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
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/util/exec"
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
		fexec := exec.NewFakeExec(t, nil)
		fexec.AddCommand("tc", "class", "show", "dev", "cbr0").
			SetCombinedOutput(test.output, test.err)
		shaper := &tcShaper{e: fexec, iface: "cbr0"}
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
		fexec.AssertExpectedCommands()
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
			err:       errors.New("test error"),
			expectErr: true,
		},
	}
	for _, test := range tests {
		fexec := exec.NewFakeExec(t, nil)
		fexec.AddCommand("tc", "filter", "show", "dev", "cbr0").
			SetCombinedOutput(test.output, test.err)
		shaper := &tcShaper{e: fexec, iface: "cbr0"}
		class, handle, found, err := shaper.findCIDRClass(test.cidr)
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
					t.Errorf("unexpectedly found an interface: %s %s", class, handle)
				}
			} else {
				if class != test.expectedClass {
					t.Errorf("expected: %s, found %s", test.expectedClass, class)
				}
				if handle != test.expectedHandle {
					t.Errorf("expected: %s, found %s", test.expectedHandle, handle)
				}
			}
		}
		fexec.AssertExpectedCommands()
	}
}

func TestGetCIDRs(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("tc", "filter", "show", "dev", "cbr0").
		SetCombinedOutput(tcFilterOutput, nil)
	shaper := &tcShaper{e: fexec, iface: "cbr0"}
	cidrs, err := shaper.GetCIDRs()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	expectedCidrs := []string{"172.17.0.2/32", "1.2.0.0/16"}
	if !reflect.DeepEqual(cidrs, expectedCidrs) {
		t.Errorf("expected: %v, saw: %v", expectedCidrs, cidrs)
	}
	fexec.AssertExpectedCommands()
}

func TestLimit(t *testing.T) {
	tests := []struct {
		cidr      string
		ingress   *resource.Quantity
		egress    *resource.Quantity
		expectErr bool
		err       error
	}{
		{
			cidr:    "1.2.3.4/32",
			ingress: resource.NewQuantity(10, resource.DecimalSI),
			egress:  resource.NewQuantity(20, resource.DecimalSI),
		},
		{
			cidr:    "1.2.3.4/32",
			ingress: resource.NewQuantity(10, resource.DecimalSI),
			egress:  nil,
		},
		{
			cidr:    "1.2.3.4/32",
			ingress: nil,
			egress:  resource.NewQuantity(20, resource.DecimalSI),
		},
		{
			cidr:    "1.2.3.4/32",
			ingress: nil,
			egress:  nil,
		},
		{
			err:       errors.New("test error"),
			ingress:   resource.NewQuantity(10, resource.DecimalSI),
			egress:    resource.NewQuantity(20, resource.DecimalSI),
			expectErr: true,
		},
	}

	for _, test := range tests {
		fexec := exec.NewFakeExec(t, nil)
		if test.err != nil {
			fexec.AddCommand("tc", "class", "show", "dev", "cbr0").
				SetCombinedOutput("", test.err)
		} else {
			nextClassShowOutput := tcClassOutput
			nextClassID := "1:5"
			if test.egress != nil {
				fexec.AddCommand("tc", "class", "show", "dev", "cbr0").
					SetCombinedOutput(nextClassShowOutput, nil)
				fexec.AddCommand("tc", "class", "add", "dev", "cbr0", "parent", "1:", "classid", nextClassID, "htb", "rate", makeKBitString(test.egress)).
					SetCombinedOutput("", nil)
				fexec.AddCommand("tc", "filter", "add", "dev", "cbr0", "protocol", "ip", "parent", "1:0", "prio", "1", "u32", "match", "ip", "dst", test.cidr, "flowid", nextClassID).
					SetCombinedOutput("", nil)

				nextClassShowOutput = tcClassOutput2
				nextClassID = "1:6"
			}
			if test.ingress != nil {
				fexec.AddCommand("tc", "class", "show", "dev", "cbr0").
					SetCombinedOutput(nextClassShowOutput, nil)
				fexec.AddCommand("tc", "class", "add", "dev", "cbr0", "parent", "1:", "classid", nextClassID, "htb", "rate", makeKBitString(test.ingress)).
					SetCombinedOutput("", nil)
				fexec.AddCommand("tc", "filter", "add", "dev", "cbr0", "protocol", "ip", "parent", "1:0", "prio", "1", "u32", "match", "ip", "src", test.cidr, "flowid", nextClassID).
					SetCombinedOutput("", nil)
			}
		}
		shaper := &tcShaper{e: fexec, iface: "cbr0"}
		if err := shaper.Limit(test.cidr, test.ingress, test.egress); err != nil && !test.expectErr {
			t.Errorf("unexpected error: %v", err)
			return
		} else if err == nil && test.expectErr {
			t.Error("unexpected non-error")
			return
		}
		fexec.AssertExpectedCommands()
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
		fexec := exec.NewFakeExec(t, nil)
		if test.err != nil {
			fexec.AddCommand("tc", "filter", "show", "dev", "cbr0").
				SetCombinedOutput("", test.err)
		} else {
			fexec.AddCommand("tc", "filter", "show", "dev", "cbr0").
				SetCombinedOutput(tcFilterOutput, nil)
			fexec.AddCommand("tc", "filter", "del", "dev", "cbr0", "parent", "1:", "proto", "ip", "prio", "1", "handle", test.expectedHandle, "u32").
				SetCombinedOutput("", test.err)
			fexec.AddCommand("tc", "class", "del", "dev", "cbr0", "parent", "1:", "classid", test.expectedClass).
				SetCombinedOutput("", test.err)
		}
		shaper := &tcShaper{e: fexec, iface: "cbr0"}

		if err := shaper.Reset(test.cidr); err != nil && !test.expectErr {
			t.Errorf("unexpected error: %v", err)
			return
		} else if test.expectErr && err == nil {
			t.Error("unexpected non-error")
			return
		}
		fexec.AssertExpectedCommands()
	}
}

var tcQdisc = "qdisc htb 1: root refcnt 2 r2q 10 default 30 direct_packets_stat 0\n"

func TestReconcileInterfaceExists(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("tc", "qdisc", "show", "dev", "cbr0").
		SetCombinedOutput(tcQdisc, nil)
	shaper := &tcShaper{e: fexec, iface: "cbr0"}
	err := shaper.ReconcileInterface()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	fexec.AssertExpectedCommands()
}

func testReconcileInterfaceHasNoData(t *testing.T, output string) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("tc", "qdisc", "show", "dev", "cbr0").
		SetCombinedOutput(output, nil)
	fexec.AddCommand("tc", "qdisc", "add", "dev", "cbr0", "root", "handle", "1:", "htb", "default", "30").
		SetCombinedOutput("", nil)
	shaper := &tcShaper{e: fexec, iface: "cbr0"}
	err := shaper.ReconcileInterface()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	fexec.AssertExpectedCommands()
}

func TestReconcileInterfaceDoesntExist(t *testing.T) {
	testReconcileInterfaceHasNoData(t, "\n")
}

var tcQdiscNoqueue = "qdisc noqueue 0: root refcnt 2 \n"

func TestReconcileInterfaceExistsWithNoqueue(t *testing.T) {
	testReconcileInterfaceHasNoData(t, tcQdiscNoqueue)
}

func TestReconcileInterfaceIsWrong(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("tc", "qdisc", "show", "dev", "cbr0").
		SetCombinedOutput("qdisc htb 2: root refcnt 2 r2q 10 default 30 direct_packets_stat 0\n", nil)
	fexec.AddCommand("tc", "qdisc", "delete", "dev", "cbr0", "root", "handle", "2:").
		SetCombinedOutput("", nil)
	fexec.AddCommand("tc", "qdisc", "add", "dev", "cbr0", "root", "handle", "1:", "htb", "default", "30").
		SetCombinedOutput("", nil)
	shaper := &tcShaper{e: fexec, iface: "cbr0"}
	err := shaper.ReconcileInterface()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	fexec.AssertExpectedCommands()

	fexec = exec.NewFakeExec(t, nil)
	fexec.AddCommand("tc", "qdisc", "show", "dev", "cbr0").
		SetCombinedOutput("qdisc foo 1: root refcnt 2 r2q 10 default 30 direct_packets_stat 0\n", nil)
	fexec.AddCommand("tc", "qdisc", "delete", "dev", "cbr0", "root", "handle", "1:").
		SetCombinedOutput("", nil)
	fexec.AddCommand("tc", "qdisc", "add", "dev", "cbr0", "root", "handle", "1:", "htb", "default", "30").
		SetCombinedOutput("", nil)
	shaper = &tcShaper{e: fexec, iface: "cbr0"}
	err = shaper.ReconcileInterface()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	fexec.AssertExpectedCommands()
}
