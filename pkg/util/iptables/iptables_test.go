/*
Copyright 2014 The Kubernetes Authors.

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

package iptables

import (
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/util/dbus"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/sets"
)

func getIPTablesCommand(protocol Protocol) string {
	if protocol == ProtocolIpv4 {
		return cmdIPTables
	}
	if protocol == ProtocolIpv6 {
		return cmdIp6tables
	}
	panic("Unknown protocol")
}

func testEnsureChain(t *testing.T, protocol Protocol) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
			// Exists.
			func() ([]byte, error) { return nil, &exec.FakeExitError{Status: 1} },
			// Failure.
			func() ([]byte, error) { return nil, &exec.FakeExitError{Status: 2} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), protocol)
	defer runner.Destroy()
	// Success.
	exists, err := runner.EnsureChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if exists {
		t.Errorf("expected exists = false")
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	cmd := getIPTablesCommand(protocol)
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll(cmd, "-t", "nat", "-N", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
	// Exists.
	exists, err = runner.EnsureChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if !exists {
		t.Errorf("expected exists = true")
	}
	// Failure.
	_, err = runner.EnsureChain(TableNAT, Chain("FOOBAR"))
	if err == nil {
		t.Errorf("expected failure")
	}
}

func TestEnsureChainIpv4(t *testing.T) {
	testEnsureChain(t, ProtocolIpv4)
}

func TestEnsureChainIpv6(t *testing.T) {
	testEnsureChain(t, ProtocolIpv6)
}

func TestFlushChain(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
			// Failure.
			func() ([]byte, error) { return nil, &exec.FakeExitError{Status: 1} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	// Success.
	err := runner.FlushChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll("iptables", "-t", "nat", "-F", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
	// Failure.
	err = runner.FlushChain(TableNAT, Chain("FOOBAR"))
	if err == nil {
		t.Errorf("expected failure")
	}
}

func TestDeleteChain(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
			// Failure.
			func() ([]byte, error) { return nil, &exec.FakeExitError{Status: 1} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	// Success.
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll("iptables", "-t", "nat", "-X", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
	// Failure.
	err = runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err == nil {
		t.Errorf("expected failure")
	}
}

func TestEnsureRuleAlreadyExists(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Success of that exec means "done".
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	exists, err := runner.EnsureRule(Append, TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if !exists {
		t.Errorf("expected exists = true")
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll("iptables", "-t", "nat", "-C", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
}

func TestEnsureRuleNew(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// Status 1 on the first call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{Status: 1} },
			// Success on the second call.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	exists, err := runner.EnsureRule(Append, TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if exists {
		t.Errorf("expected exists = false")
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll("iptables", "-t", "nat", "-A", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
}

func TestEnsureRuleErrorChecking(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// Status 2 on the first call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{Status: 2} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	_, err := runner.EnsureRule(Append, TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
}

func TestEnsureRuleErrorCreating(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// Status 1 on the first call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{Status: 1} },
			// Status 1 on the second call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{Status: 1} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	_, err := runner.EnsureRule(Append, TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
}

func TestDeleteRuleAlreadyExists(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// Status 1 on the first call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{Status: 1} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Failure of that exec means "does not exist".
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll("iptables", "-t", "nat", "-C", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
}

func TestDeleteRuleNew(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// Success on the first call.
			func() ([]byte, error) { return []byte{}, nil },
			// Success on the second call.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Success of that means delete it.
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll("iptables", "-t", "nat", "-D", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
}

func TestDeleteRuleErrorChecking(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// Status 2 on the first call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{Status: 2} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
}

func TestDeleteRuleErrorCreating(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// Success on the first call.
			func() ([]byte, error) { return []byte{}, nil },
			// Status 1 on the second call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{Status: 1} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Success of that means delete it.
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
}

func TestGetIPTablesHasCheckCommand(t *testing.T) {
	testCases := []struct {
		Version  string
		Err      bool
		Expected bool
	}{
		{"iptables v1.4.7", false, false},
		{"iptables v1.4.11", false, true},
		{"iptables v1.4.19.1", false, true},
		{"iptables v2.0.0", false, true},
		{"total junk", true, false},
	}

	for _, testCase := range testCases {
		fcmd := exec.FakeCmd{
			CombinedOutputScript: []exec.FakeCombinedOutputAction{
				func() ([]byte, error) { return []byte(testCase.Version), nil },
			},
		}
		fexec := exec.FakeExec{
			CommandScript: []exec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			},
		}
		version, err := getIPTablesVersionString(&fexec)
		if (err != nil) != testCase.Err {
			t.Errorf("Expected error: %v, Got error: %v", testCase.Err, err)
		}
		if err == nil {
			check := getIPTablesHasCheckCommand(version)
			if testCase.Expected != check {
				t.Errorf("Expected result: %v, Got result: %v", testCase.Expected, check)
			}
		}
	}
}

func TestCheckRuleWithoutCheckPresent(t *testing.T) {
	iptables_save_output := `# Generated by iptables-save v1.4.7 on Wed Oct 29 14:56:01 2014
*nat
:PREROUTING ACCEPT [2136997:197881818]
:POSTROUTING ACCEPT [4284525:258542680]
:OUTPUT ACCEPT [5901660:357267963]
-A PREROUTING -m addrtype --dst-type LOCAL -j DOCKER
COMMIT
# Completed on Wed Oct 29 14:56:01 2014`

	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// Success.
			func() ([]byte, error) { return []byte(iptables_save_output), nil },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// The first Command() call is checking the rule.  Success of that exec means "done".
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := &runner{exec: &fexec}
	exists, err := runner.checkRuleWithoutCheck(TableNAT, ChainPrerouting, "-m", "addrtype", "-j", "DOCKER", "--dst-type", "LOCAL")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if !exists {
		t.Errorf("expected exists = true")
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[0]...).HasAll("iptables-save", "-t", "nat") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
	}
}

func TestCheckRuleWithoutCheckAbsent(t *testing.T) {
	iptables_save_output := `# Generated by iptables-save v1.4.7 on Wed Oct 29 14:56:01 2014
*nat
:PREROUTING ACCEPT [2136997:197881818]
:POSTROUTING ACCEPT [4284525:258542680]
:OUTPUT ACCEPT [5901660:357267963]
-A PREROUTING -m addrtype --dst-type LOCAL -j DOCKER
COMMIT
# Completed on Wed Oct 29 14:56:01 2014`

	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// Success.
			func() ([]byte, error) { return []byte(iptables_save_output), nil },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// The first Command() call is checking the rule.  Success of that exec means "done".
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := &runner{exec: &fexec}
	exists, err := runner.checkRuleWithoutCheck(TableNAT, ChainPrerouting, "-m", "addrtype", "-j", "DOCKER")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if exists {
		t.Errorf("expected exists = false")
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[0]...).HasAll("iptables-save", "-t", "nat") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
	}
}

func TestIPTablesWaitFlag(t *testing.T) {
	testCases := []struct {
		Version string
		Result  string
	}{
		{"0.55.55", ""},
		{"1.0.55", ""},
		{"1.4.19", ""},
		{"1.4.20", "-w"},
		{"1.4.21", "-w"},
		{"1.4.22", "-w2"},
		{"1.5.0", "-w2"},
		{"2.0.0", "-w2"},
	}

	for _, testCase := range testCases {
		result := getIPTablesWaitFlag(testCase.Version)
		if strings.Join(result, "") != testCase.Result {
			t.Errorf("For %s expected %v got %v", testCase.Version, testCase.Result, result)
		}
	}
}

func TestWaitFlagUnavailable(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.4.19"), nil },
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if sets.NewString(fcmd.CombinedOutputLog[1]...).HasAny("-w", "-w2") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
}

func TestWaitFlagOld(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.4.20"), nil },
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll("iptables", "-w") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
	if sets.NewString(fcmd.CombinedOutputLog[1]...).HasAny("-w2") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
}

func TestWaitFlagNew(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.4.22"), nil },
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll("iptables", "-w2") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
	if sets.NewString(fcmd.CombinedOutputLog[1]...).HasAny("-w") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
}

func TestReload(t *testing.T) {
	dbusConn := dbus.NewFakeConnection()
	dbusConn.SetBusObject(func(method string, args ...interface{}) ([]interface{}, error) { return nil, nil })
	dbusConn.AddObject(firewalldName, firewalldPath, func(method string, args ...interface{}) ([]interface{}, error) { return nil, nil })
	fdbus := dbus.NewFake(dbusConn, nil)

	reloaded := make(chan bool, 2)

	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.4.22"), nil },

			// first reload
			// EnsureChain
			func() ([]byte, error) { return []byte{}, nil },
			// EnsureRule abc check
			func() ([]byte, error) { return []byte{}, &exec.FakeExitError{Status: 1} },
			// EnsureRule abc
			func() ([]byte, error) { return []byte{}, nil },

			// second reload
			// EnsureChain
			func() ([]byte, error) { return []byte{}, nil },
			// EnsureRule abc check
			func() ([]byte, error) { return []byte{}, &exec.FakeExitError{Status: 1} },
			// EnsureRule abc
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}

	runner := New(&fexec, fdbus, ProtocolIpv4)
	defer runner.Destroy()

	runner.AddReloadFunc(func() {
		exists, err := runner.EnsureChain(TableNAT, Chain("FOOBAR"))
		if err != nil {
			t.Errorf("expected success, got %v", err)
		}
		if exists {
			t.Errorf("expected exists = false")
		}
		reloaded <- true
	})

	runner.AddReloadFunc(func() {
		exists, err := runner.EnsureRule(Append, TableNAT, ChainOutput, "abc", "123")
		if err != nil {
			t.Errorf("expected success, got %v", err)
		}
		if exists {
			t.Errorf("expected exists = false")
		}
		reloaded <- true
	})

	dbusConn.EmitSignal("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "NameOwnerChanged", firewalldName, "", ":1.1")
	<-reloaded
	<-reloaded

	if fcmd.CombinedOutputCalls != 4 {
		t.Errorf("expected 4 CombinedOutput() calls total, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll("iptables", "-t", "nat", "-N", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll("iptables", "-t", "nat", "-C", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
	if !sets.NewString(fcmd.CombinedOutputLog[3]...).HasAll("iptables", "-t", "nat", "-A", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[3])
	}

	go func() { time.Sleep(time.Second / 100); reloaded <- true }()
	dbusConn.EmitSignal(firewalldName, firewalldPath, firewalldInterface, "DefaultZoneChanged", "public")
	dbusConn.EmitSignal("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "NameOwnerChanged", "io.k8s.Something", "", ":1.1")
	<-reloaded

	if fcmd.CombinedOutputCalls != 4 {
		t.Errorf("Incorrect signal caused a reload")
	}

	dbusConn.EmitSignal(firewalldName, firewalldPath, firewalldInterface, "Reloaded")
	<-reloaded
	<-reloaded

	if fcmd.CombinedOutputCalls != 7 {
		t.Errorf("expected 7 CombinedOutput() calls total, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[4]...).HasAll("iptables", "-t", "nat", "-N", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[4])
	}
	if !sets.NewString(fcmd.CombinedOutputLog[5]...).HasAll("iptables", "-t", "nat", "-C", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[5])
	}
	if !sets.NewString(fcmd.CombinedOutputLog[6]...).HasAll("iptables", "-t", "nat", "-A", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[6])
	}
}
