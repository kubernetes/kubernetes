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

package util

import (
	"fmt"
	"strconv"
	"strings"
	"testing"

	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

func TestExecConntrackTool(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			func() ([]byte, error) { return []byte("1 flow entries have been deleted"), nil },
			func() ([]byte, error) { return []byte("1 flow entries have been deleted"), nil },
			func() ([]byte, error) {
				return []byte(""), fmt.Errorf("conntrack v1.4.2 (conntrack-tools): 0 flow entries have been deleted.")
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
		err := ExecConntrackTool(&fexec, testCases[i]...)

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
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			func() ([]byte, error) { return []byte("1 flow entries have been deleted"), nil },
			func() ([]byte, error) { return []byte("1 flow entries have been deleted"), nil },
			func() ([]byte, error) {
				return []byte(""), fmt.Errorf("conntrack v1.4.2 (conntrack-tools): 0 flow entries have been deleted.")
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
		{
			"10.240.0.3",
			"10.240.0.5",
		},
		{
			"10.240.0.4",
		},
	}

	svcCount := 0
	for i := range testCases {
		for _, ip := range testCases[i] {
			if err := ClearUDPConntrackForIP(&fexec, ip); err != nil {
				t.Errorf("Unexepected error: %v", err)
			}
			expectCommand := fmt.Sprintf("conntrack -D --orig-dst %s -p udp", ip)
			execCommand := strings.Join(fcmd.CombinedOutputLog[svcCount], " ")
			if expectCommand != execCommand {
				t.Errorf("Exepect comand: %s, but executed %s", expectCommand, execCommand)
			}
			svcCount += 1
		}
		if svcCount != fexec.CommandCalls {
			t.Errorf("Exepect comand executed %d times, but got %d", svcCount, fexec.CommandCalls)
		}
	}
}

func TestClearUDPConntrackForPort(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			func() ([]byte, error) { return []byte("1 flow entries have been deleted"), nil },
			func() ([]byte, error) {
				return []byte(""), fmt.Errorf("conntrack v1.4.2 (conntrack-tools): 0 flow entries have been deleted.")
			},
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return cmd, nil },
	}

	testCases := []string{
		"8080",
		"9090",
	}
	svcCount := 0
	for i := range testCases {
		portNum, _ := strconv.Atoi(testCases[i])
		err := ClearUDPConntrackForPort(&fexec, portNum)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		expectCommand := fmt.Sprintf("conntrack -D -p udp --dport %s", testCases[i])
		execCommand := strings.Join(fcmd.CombinedOutputLog[svcCount], " ")
		if expectCommand != execCommand {
			t.Errorf("Exepect comand: %s, but executed %s", expectCommand, execCommand)
		}
		svcCount += 1

		if svcCount != fexec.CommandCalls {
			t.Errorf("Exepect comand executed %d times, but got %d", svcCount, fexec.CommandCalls)
		}
	}
}

func TestDeleteUDPConnections(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			func() ([]byte, error) { return []byte("1 flow entries have been deleted"), nil },
			func() ([]byte, error) {
				return []byte(""), fmt.Errorf("conntrack v1.4.2 (conntrack-tools): 0 flow entries have been deleted.")
			},
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return cmd, nil },
	}

	testCases := []struct {
		origin string
		dest   string
	}{
		{
			origin: "1.2.3.4",
			dest:   "10.20.30.40",
		},
		{
			origin: "2.3.4.5",
			dest:   "20.30.40.50",
		},
	}
	svcCount := 0
	for i := range testCases {
		err := ClearUDPConntrackForPeers(&fexec, testCases[i].origin, testCases[i].dest)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		expectCommand := fmt.Sprintf("conntrack -D --orig-dst %s --dst-nat %s -p udp", testCases[i].origin, testCases[i].dest)
		execCommand := strings.Join(fcmd.CombinedOutputLog[svcCount], " ")
		if expectCommand != execCommand {
			t.Errorf("Exepect comand: %s, but executed %s", expectCommand, execCommand)
		}
		svcCount += 1

		if svcCount != fexec.CommandCalls {
			t.Errorf("Exepect comand executed %d times, but got %d", svcCount, fexec.CommandCalls)
		}
	}
}
