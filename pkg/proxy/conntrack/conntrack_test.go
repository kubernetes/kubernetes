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

package conntrack

import (
	"fmt"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

var success = func() ([]byte, []byte, error) { return []byte("1 flow entries have been deleted"), nil, nil }
var nothingToDelete = func() ([]byte, []byte, error) {
	return []byte(""), nil, fmt.Errorf("conntrack v1.4.2 (conntrack-tools): 0 flow entries have been deleted")
}

func makeCT(result fakeexec.FakeAction) (*fakeexec.FakeExec, *fakeexec.FakeCmd) {
	fcmd := &fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{result},
	}
	fexec := &fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return cmd, nil },
	}

	return fexec, fcmd
}

// Gets the command that fexec executed. (If it didn't execute any commands, this will
// return "".)
func getExecutedCommand(fexec *fakeexec.FakeExec, fcmd *fakeexec.FakeCmd) string {
	// FakeExec panics if you try to run more commands than you set it up for. So the
	// only possibilities here are that we ran 1 command or we ran 0.
	if fexec.CommandCalls != 1 {
		return ""
	}
	return strings.Join(fcmd.CombinedOutputLog[0], " ")
}

func TestExec(t *testing.T) {
	testCases := []struct {
		args      []string
		result    fakeexec.FakeAction
		expectErr bool
	}{
		{
			args:      []string{"-D", "-p", "udp", "-d", "10.0.240.1"},
			result:    success,
			expectErr: false,
		},
		{
			args:      []string{"-D", "-p", "udp", "--orig-dst", "10.240.0.2", "--dst-nat", "10.0.10.2"},
			result:    nothingToDelete,
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		fexec, fcmd := makeCT(tc.result)
		err := Exec(fexec, tc.args...)
		if tc.expectErr {
			if err == nil {
				t.Errorf("expected err, got %v", err)
			}
		} else {
			if err != nil {
				t.Errorf("expected success, got %v", err)
			}
		}

		execCmd := getExecutedCommand(fexec, fcmd)
		expectCmd := "conntrack " + strings.Join(tc.args, " ")
		if execCmd != expectCmd {
			t.Errorf("expect execute command: %s, but got: %s", expectCmd, execCmd)
		}
	}
}

func TestClearEntriesForIP(t *testing.T) {
	testCases := []struct {
		name string
		ip   string

		expectCommand string
	}{
		{
			name: "IPv4",
			ip:   "10.240.0.3",

			expectCommand: "conntrack -D --orig-dst 10.240.0.3 -p udp",
		},
		{
			name: "IPv6",
			ip:   "2001:db8::10",

			expectCommand: "conntrack -D --orig-dst 2001:db8::10 -p udp -f ipv6",
		},
	}

	for _, tc := range testCases {
		fexec, fcmd := makeCT(success)
		if err := ClearEntriesForIP(fexec, tc.ip, v1.ProtocolUDP); err != nil {
			t.Errorf("%s/success: Unexpected error: %v", tc.name, err)
		}
		execCommand := getExecutedCommand(fexec, fcmd)
		if tc.expectCommand != execCommand {
			t.Errorf("%s/success: Expect command: %s, but executed %s", tc.name, tc.expectCommand, execCommand)
		}

		fexec, _ = makeCT(nothingToDelete)
		if err := ClearEntriesForIP(fexec, tc.ip, v1.ProtocolUDP); err != nil {
			t.Errorf("%s/nothing to delete: Unexpected error: %v", tc.name, err)
		}
	}
}

func TestClearEntriesForPort(t *testing.T) {
	testCases := []struct {
		name   string
		port   int
		isIPv6 bool

		expectCommand string
	}{
		{
			name:   "IPv4",
			port:   8080,
			isIPv6: false,

			expectCommand: "conntrack -D -p udp --dport 8080",
		},
		{
			name:   "IPv6",
			port:   6666,
			isIPv6: true,

			expectCommand: "conntrack -D -p udp --dport 6666 -f ipv6",
		},
	}

	for _, tc := range testCases {
		fexec, fcmd := makeCT(success)
		err := ClearEntriesForPort(fexec, tc.port, tc.isIPv6, v1.ProtocolUDP)
		if err != nil {
			t.Errorf("%s/success: Unexpected error: %v", tc.name, err)
		}
		execCommand := getExecutedCommand(fexec, fcmd)
		if tc.expectCommand != execCommand {
			t.Errorf("%s/success: Expect command: %s, but executed %s", tc.name, tc.expectCommand, execCommand)
		}

		fexec, _ = makeCT(nothingToDelete)
		err = ClearEntriesForPort(fexec, tc.port, tc.isIPv6, v1.ProtocolUDP)
		if err != nil {
			t.Errorf("%s/nothing to delete: Unexpected error: %v", tc.name, err)
		}
	}
}

func TestClearEntriesForNAT(t *testing.T) {
	testCases := []struct {
		name   string
		origin string
		dest   string

		expectCommand string
	}{
		{
			name:   "IPv4",
			origin: "1.2.3.4",
			dest:   "10.20.30.40",

			expectCommand: "conntrack -D --orig-dst 1.2.3.4 --dst-nat 10.20.30.40 -p udp",
		},
		{
			name:   "IPv6",
			origin: "fd00::600d:f00d",
			dest:   "2001:db8::5",

			expectCommand: "conntrack -D --orig-dst fd00::600d:f00d --dst-nat 2001:db8::5 -p udp -f ipv6",
		},
	}

	for _, tc := range testCases {
		fexec, fcmd := makeCT(success)
		err := ClearEntriesForNAT(fexec, tc.origin, tc.dest, v1.ProtocolUDP)
		if err != nil {
			t.Errorf("%s/success: unexpected error: %v", tc.name, err)
		}
		execCommand := getExecutedCommand(fexec, fcmd)
		if tc.expectCommand != execCommand {
			t.Errorf("%s/success: Expect command: %s, but executed %s", tc.name, tc.expectCommand, execCommand)
		}

		fexec, _ = makeCT(nothingToDelete)
		err = ClearEntriesForNAT(fexec, tc.origin, tc.dest, v1.ProtocolUDP)
		if err != nil {
			t.Errorf("%s/nothing to delete: unexpected error: %v", tc.name, err)
		}
	}
}

func TestClearEntriesForPortNAT(t *testing.T) {
	testCases := []struct {
		name string
		port int
		dest string

		expectCommand string
	}{
		{
			name: "IPv4",
			port: 30211,
			dest: "1.2.3.4",

			expectCommand: "conntrack -D -p udp --dport 30211 --dst-nat 1.2.3.4",
		},
		{
			name: "IPv6",
			port: 30212,
			dest: "2600:5200::7800",

			expectCommand: "conntrack -D -p udp --dport 30212 --dst-nat 2600:5200::7800 -f ipv6",
		},
	}

	for _, tc := range testCases {
		fexec, fcmd := makeCT(success)
		err := ClearEntriesForPortNAT(fexec, tc.dest, tc.port, v1.ProtocolUDP)
		if err != nil {
			t.Errorf("%s/success: unexpected error: %v", tc.name, err)
		}
		execCommand := getExecutedCommand(fexec, fcmd)
		if tc.expectCommand != execCommand {
			t.Errorf("%s/success: Expect command: %s, but executed %s", tc.name, tc.expectCommand, execCommand)
		}

		fexec, _ = makeCT(nothingToDelete)
		err = ClearEntriesForPortNAT(fexec, tc.dest, tc.port, v1.ProtocolUDP)
		if err != nil {
			t.Errorf("%s/nothing to delete: unexpected error: %v", tc.name, err)
		}
	}
}
