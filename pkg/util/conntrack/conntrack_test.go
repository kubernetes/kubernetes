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

package conntrack

import (
	"fmt"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
	utilnet "k8s.io/utils/net"
)

func familyParamStr(isIPv6 bool) string {
	if isIPv6 {
		return " -f ipv6"
	}
	return ""
}

func TestExecConntrackTool(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return []byte("1 flow entries have been deleted"), nil, nil },
			func() ([]byte, []byte, error) { return []byte("1 flow entries have been deleted"), nil, nil },
			func() ([]byte, []byte, error) {
				return []byte(""), nil, fmt.Errorf("conntrack v1.4.2 (conntrack-tools): 0 flow entries have been deleted")
			},
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return cmd, nil },
	}

	testCases := [][]string{
		{"-L", "-p", "udp"},
		{"-D", "-p", "udp", "-d", "10.0.240.1"},
		{"-D", "-p", "udp", "--orig-dst", "10.240.0.2", "--dst-nat", "10.0.10.2"},
	}

	expectErr := []bool{false, false, true}

	for i := range testCases {
		err := Exec(&fexec, testCases[i]...)

		if expectErr[i] {
			if err == nil {
				t.Errorf("expected err, got %v", err)
			}
		} else {
			if err != nil {
				t.Errorf("expected success, got %v", err)
			}
		}

		execCmd := strings.Join(fcmd.CombinedOutputLog[i], " ")
		expectCmd := fmt.Sprintf("%s %s", "conntrack", strings.Join(testCases[i], " "))

		if execCmd != expectCmd {
			t.Errorf("expect execute command: %s, but got: %s", expectCmd, execCmd)
		}
	}
}

func TestClearUDPConntrackForIP(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return []byte("1 flow entries have been deleted"), nil, nil },
			func() ([]byte, []byte, error) { return []byte("1 flow entries have been deleted"), nil, nil },
			func() ([]byte, []byte, error) {
				return []byte(""), nil, fmt.Errorf("conntrack v1.4.2 (conntrack-tools): 0 flow entries have been deleted")
			},
			func() ([]byte, []byte, error) { return []byte("1 flow entries have been deleted"), nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return cmd, nil },
	}

	testCases := []struct {
		name string
		ip   string
	}{
		{"IPv4 success", "10.240.0.3"},
		{"IPv4 success", "10.240.0.5"},
		{"IPv4 simulated error", "10.240.0.4"},
		{"IPv6 success", "2001:db8::10"},
	}

	svcCount := 0
	for _, tc := range testCases {
		if err := ClearEntriesForIP(&fexec, tc.ip, v1.ProtocolUDP); err != nil {
			t.Errorf("%s test case:, Unexpected error: %v", tc.name, err)
		}
		expectCommand := fmt.Sprintf("conntrack -D --orig-dst %s -p udp", tc.ip) + familyParamStr(utilnet.IsIPv6String(tc.ip))
		execCommand := strings.Join(fcmd.CombinedOutputLog[svcCount], " ")
		if expectCommand != execCommand {
			t.Errorf("%s test case: Expect command: %s, but executed %s", tc.name, expectCommand, execCommand)
		}
		svcCount++
	}
	if svcCount != fexec.CommandCalls {
		t.Errorf("Expect command executed %d times, but got %d", svcCount, fexec.CommandCalls)
	}
}

func TestClearUDPConntrackForPort(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return []byte("1 flow entries have been deleted"), nil, nil },
			func() ([]byte, []byte, error) {
				return []byte(""), nil, fmt.Errorf("conntrack v1.4.2 (conntrack-tools): 0 flow entries have been deleted")
			},
			func() ([]byte, []byte, error) { return []byte("1 flow entries have been deleted"), nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return cmd, nil },
	}

	testCases := []struct {
		name   string
		port   int
		isIPv6 bool
	}{
		{"IPv4, no error", 8080, false},
		{"IPv4, simulated error", 9090, false},
		{"IPv6, no error", 6666, true},
	}
	svcCount := 0
	for _, tc := range testCases {
		err := ClearEntriesForPort(&fexec, tc.port, tc.isIPv6, v1.ProtocolUDP)
		if err != nil {
			t.Errorf("%s test case: Unexpected error: %v", tc.name, err)
		}
		expectCommand := fmt.Sprintf("conntrack -D -p udp --dport %d", tc.port) + familyParamStr(tc.isIPv6)
		execCommand := strings.Join(fcmd.CombinedOutputLog[svcCount], " ")
		if expectCommand != execCommand {
			t.Errorf("%s test case: Expect command: %s, but executed %s", tc.name, expectCommand, execCommand)
		}
		svcCount++
	}
	if svcCount != fexec.CommandCalls {
		t.Errorf("Expect command executed %d times, but got %d", svcCount, fexec.CommandCalls)
	}
}

func TestDeleteUDPConnections(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return []byte("1 flow entries have been deleted"), nil, nil },
			func() ([]byte, []byte, error) {
				return []byte(""), nil, fmt.Errorf("conntrack v1.4.2 (conntrack-tools): 0 flow entries have been deleted")
			},
			func() ([]byte, []byte, error) { return []byte("1 flow entries have been deleted"), nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return cmd, nil },
	}

	testCases := []struct {
		name   string
		origin string
		dest   string
	}{
		{
			name:   "IPv4 success",
			origin: "1.2.3.4",
			dest:   "10.20.30.40",
		},
		{
			name:   "IPv4 simulated failure",
			origin: "2.3.4.5",
			dest:   "20.30.40.50",
		},
		{
			name:   "IPv6 success",
			origin: "fd00::600d:f00d",
			dest:   "2001:db8::5",
		},
	}
	svcCount := 0
	for i, tc := range testCases {
		err := ClearEntriesForNAT(&fexec, tc.origin, tc.dest, v1.ProtocolUDP)
		if err != nil {
			t.Errorf("%s test case: unexpected error: %v", tc.name, err)
		}
		expectCommand := fmt.Sprintf("conntrack -D --orig-dst %s --dst-nat %s -p udp", tc.origin, tc.dest) + familyParamStr(utilnet.IsIPv6String(tc.origin))
		execCommand := strings.Join(fcmd.CombinedOutputLog[i], " ")
		if expectCommand != execCommand {
			t.Errorf("%s test case: Expect command: %s, but executed %s", tc.name, expectCommand, execCommand)
		}
		svcCount++
	}
	if svcCount != fexec.CommandCalls {
		t.Errorf("Expect command executed %d times, but got %d", svcCount, fexec.CommandCalls)
	}
}

func TestClearUDPConntrackForPortNAT(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return []byte("1 flow entries have been deleted"), nil, nil },
			func() ([]byte, []byte, error) {
				return []byte(""), nil, fmt.Errorf("conntrack v1.4.2 (conntrack-tools): 0 flow entries have been deleted")
			},
			func() ([]byte, []byte, error) { return []byte("1 flow entries have been deleted"), nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return cmd, nil },
	}
	testCases := []struct {
		name string
		port int
		dest string
	}{
		{
			name: "IPv4 success",
			port: 30211,
			dest: "1.2.3.4",
		},
	}
	svcCount := 0
	for i, tc := range testCases {
		err := ClearEntriesForPortNAT(&fexec, tc.dest, tc.port, v1.ProtocolUDP)
		if err != nil {
			t.Errorf("%s test case: unexpected error: %v", tc.name, err)
		}
		expectCommand := fmt.Sprintf("conntrack -D -p udp --dport %d --dst-nat %s", tc.port, tc.dest) + familyParamStr(utilnet.IsIPv6String(tc.dest))
		execCommand := strings.Join(fcmd.CombinedOutputLog[i], " ")
		if expectCommand != execCommand {
			t.Errorf("%s test case: Expect command: %s, but executed %s", tc.name, expectCommand, execCommand)
		}
		svcCount++
	}
	if svcCount != fexec.CommandCalls {
		t.Errorf("Expect command executed %d times, but got %d", svcCount, fexec.CommandCalls)
	}
}
