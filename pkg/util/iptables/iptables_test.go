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
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/dbus"
	"k8s.io/kubernetes/pkg/util/exec"
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
-A PREROUTING -m addrtype --dst-type LOCAL -m mark --mark 0x00004000/0x00004000 -j DOCKER
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
	exists, err := runner.checkRuleWithoutCheck(
		TableNAT, ChainPrerouting,
		"-m", "addrtype",
		"-m", "mark", "--mark", "0x4000/0x4000",
		"-j", "DOCKER",
		"--dst-type", "LOCAL")
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

func TestParseArgsSplitLine(t *testing.T) {
	testCases := []struct {
		line string
		args []string
		err  string
	}{
		{
			line: "-A FORWARD_IN_ZONES -i tun0 -g FWDI_FedoraWorkstation",
			args: []string{"-A", "FORWARD_IN_ZONES", "-i", "tun0", "-g", "FWDI_FedoraWorkstation"},
		},
		{
			line: "-A PREROUTING --comment \"last rule in the chain\" -j KUBE-NODEPORT-CONTAINER",
			args: []string{"-A", "PREROUTING", "--comment", "last rule in the chain", "-j", "KUBE-NODEPORT-CONTAINER"},
		},
		{
			line: "-A PREROUTING --comment \"default/kubernetes:dns\" -j KUBE-MARK-MASQ",
			args: []string{"-A", "PREROUTING", "--comment", "default/kubernetes:dns", "-j", "KUBE-MARK-MASQ"},
		},
		{
			line: "-A PREROUTING --comment \"\" -j KUBE-MARK-MASQ",
			args: []string{"-A", "PREROUTING", "--comment", "", "-j", "KUBE-MARK-MASQ"},
		},
		{
			line: "-A PREROUTING --comment \"\\'\\\\\" -j KUBE-MARK-MASQ",
			args: []string{"-A", "PREROUTING", "--comment", "'\\", "-j", "KUBE-MARK-MASQ"},
		},
		{
			line: "-A PREROUTING --comment \"foo \\\"bar\\\"\" -j KUBE-MARK-MASQ",
			args: []string{"-A", "PREROUTING", "--comment", "foo \"bar\"", "-j", "KUBE-MARK-MASQ"},
		},
		{
			line: "-A PREROUTING --comment \"",
			err:  "invalid iptables rule (unterminated quote)",
		},
		{
			line: "-A PREROUTING --comment \"asdfasdfasfd",
			err:  "invalid iptables rule (unterminated quote)",
		},
		{
			line: "-A PREROUTING --comment \"asdfasdfasfd asdfasdf",
			err:  "invalid iptables rule (unterminated quote)",
		},
		{
			line: "-A PREROUTING --comment \"\\",
			err:  "invalid iptables rule (not enough characters to unescape)",
		},
	}

	for _, tc := range testCases {
		args, err := splitLine(tc.line)
		if err != nil {
			if tc.err != "" {
				if !strings.HasPrefix(err.Error(), tc.err) {
					t.Errorf("expected error %v, got %v", tc.err, err)
				}
			} else {
				t.Errorf("unexpected error for line '%s': %v", tc.line, err)
			}
		} else {
			if !reflect.DeepEqual(args, tc.args) {
				t.Errorf("unexpected parse result for '%s': expected %v but got %v", tc.line, tc.args, args)
			}
		}
	}
}

func TestParseTableAddRules(t *testing.T) {
	testCases := []struct {
		table               Table
		filterChains        []Chain
		filterChainPrefixes []string
		data                []byte
		result              map[Chain][]Rule
		err                 string
	}{
		{
			table: TableNAT,
			data: []byte(`# Generated by iptables-save v1.4.21 on Wed Oct 26 02:29:33 2016
*nat
:PREROUTING ACCEPT [0:0]
:INPUT ACCEPT [0:0]
:OUTPUT ACCEPT [0:0]
:POSTROUTING ACCEPT [0:0]
:DOCKER - [0:0]
:KUBE-MARK-DROP - [0:0]
:KUBE-MARK-MASQ - [0:0]
:KUBE-POSTROUTING - [0:0]
:KUBE-SEP-CKTKXEMIKRIIY55M - [0:0]
:KUBE-SEP-EZ5ESXJRZ36JV4D4 - [0:0]
:KUBE-SEP-PATXOTJBHFPU4CNS - [0:0]
:KUBE-SERVICES - [0:0]
:KUBE-SVC-3VQ6B3MLH7E2SZT4 - [0:0]
:KUBE-SVC-BA6I5HTZKAAAJT56 - [0:0]
:KUBE-SVC-NPX46M4PTMTKRN6Y - [0:0]
-A PREROUTING -m comment --comment "kubernetes service portals" -j KUBE-SERVICES
-A PREROUTING -m addrtype --dst-type LOCAL -j DOCKER
-A PREROUTING -m addrtype --dst-type LOCAL -m comment --comment "handle service NodePorts; NOTE: this must be the last rule in the chain" -j KUBE-NODEPORT-CONTAINER
-A OUTPUT -m comment --comment "kubernetes service portals" -j KUBE-SERVICES
-A OUTPUT -m comment --comment "handle ClusterIPs; NOTE: this must be before the NodePort rules" -j KUBE-PORTALS-HOST
-A OUTPUT ! -d 127.0.0.0/8 -m addrtype --dst-type LOCAL -j DOCKER
-A OUTPUT -m addrtype --dst-type LOCAL -m comment --comment "handle service NodePorts; NOTE: this must be the last rule in the chain" -j KUBE-NODEPORT-HOST
-A POSTROUTING -m comment --comment "kubernetes postrouting rules" -j KUBE-POSTROUTING
-A POSTROUTING -s 10.128.0.0/14 -j MASQUERADE
-A POSTROUTING -s 172.18.0.0/16 ! -o docker0 -j MASQUERADE
-A DOCKER -i docker0 -j RETURN
-A KUBE-MARK-DROP -j MARK --set-xmark 0x8000/0x8000
-A KUBE-MARK-MASQ -j MARK --set-xmark 0x4000/0x4000
-A KUBE-POSTROUTING -m comment --comment "kubernetes service traffic requiring SNAT" -m mark --mark 0x4000/0x4000 -j MASQUERADE
-A KUBE-SEP-CKTKXEMIKRIIY55M -s 172.17.0.2/32 -m comment --comment "default/kubernetes:dns" -j KUBE-MARK-MASQ
-A KUBE-SEP-CKTKXEMIKRIIY55M -p udp -m comment --comment "default/kubernetes:dns" -m recent --set --name KUBE-SEP-CKTKXEMIKRIIY55M --mask 255.255.255.255 --rsource -m udp -j DNAT --to-destination 172.17.0.2:8053
-A KUBE-SEP-EZ5ESXJRZ36JV4D4 -s 172.17.0.2/32 -m comment --comment "default/kubernetes:dns-tcp" -j KUBE-MARK-MASQ
-A KUBE-SEP-EZ5ESXJRZ36JV4D4 -p tcp -m comment --comment "default/kubernetes:dns-tcp" -m recent --set --name KUBE-SEP-EZ5ESXJRZ36JV4D4 --mask 255.255.255.255 --rsource -m tcp -j DNAT --to-destination 172.17.0.2:8053
-A KUBE-SEP-PATXOTJBHFPU4CNS -s 172.17.0.2/32 -m comment --comment "default/kubernetes:https" -j KUBE-MARK-MASQ
-A KUBE-SEP-PATXOTJBHFPU4CNS -p tcp -m comment --comment "default/kubernetes:https" -m recent --set --name KUBE-SEP-PATXOTJBHFPU4CNS --mask 255.255.255.255 --rsource -m tcp -j DNAT --to-destination 172.17.0.2:8443
-A KUBE-SERVICES -d 172.30.0.1/32 -p tcp -m comment --comment "default/kubernetes:https cluster IP" -m tcp --dport 443 -j KUBE-SVC-NPX46M4PTMTKRN6Y
-A KUBE-SERVICES -d 172.30.0.1/32 -p udp -m comment --comment "default/kubernetes:dns cluster IP" -m udp --dport 53 -j KUBE-SVC-3VQ6B3MLH7E2SZT4
-A KUBE-SERVICES -d 172.30.0.1/32 -p tcp -m comment --comment "default/kubernetes:dns-tcp cluster IP" -m tcp --dport 53 -j KUBE-SVC-BA6I5HTZKAAAJT56
-A KUBE-SERVICES -d 172.46.148.244/32 -p tcp -m comment --comment "default/nginxservice: external IP" -m tcp --dport 82 -m physdev ! --physdev-is-in -m addrtype ! --src-type LOCAL -j KUBE-SVC-URRHIARQWDHNXJTW
-A KUBE-SERVICES -d 172.46.148.244/32 -p tcp -m comment --comment "default/nginxservice: external IP" -m tcp --dport 82 -m addrtype --dst-type LOCAL -j KUBE-SVC-URRHIARQWDHNXJTW
-A KUBE-SERVICES -m comment --comment "kubernetes service nodeports; NOTE: this must be the last rule in this chain" -m addrtype --dst-type LOCAL -j KUBE-NODEPORTS
-A KUBE-SVC-3VQ6B3MLH7E2SZT4 -m comment --comment "default/kubernetes:dns" -m recent --rcheck --seconds 180 --reap --name KUBE-SEP-CKTKXEMIKRIIY55M --mask 255.255.255.255 --rsource -j KUBE-SEP-CKTKXEMIKRIIY55M
-A KUBE-SVC-3VQ6B3MLH7E2SZT4 -m comment --comment "default/kubernetes:dns" -j KUBE-SEP-CKTKXEMIKRIIY55M
-A KUBE-SVC-BA6I5HTZKAAAJT56 -m comment --comment "default/kubernetes:dns-tcp" -m recent --rcheck --seconds 180 --reap --name KUBE-SEP-EZ5ESXJRZ36JV4D4 --mask 255.255.255.255 --rsource -j KUBE-SEP-EZ5ESXJRZ36JV4D4
-A KUBE-SVC-BA6I5HTZKAAAJT56 -m comment --comment "default/kubernetes:dns-tcp" -j KUBE-SEP-EZ5ESXJRZ36JV4D4
-A KUBE-SVC-NPX46M4PTMTKRN6Y -m comment --comment "default/kubernetes:https" -m recent --rcheck --seconds 180 --reap --name KUBE-SEP-PATXOTJBHFPU4CNS --mask 255.255.255.255 --rsource -j KUBE-SEP-PATXOTJBHFPU4CNS
-A KUBE-SVC-NPX46M4PTMTKRN6Y -m comment --comment "default/kubernetes:https" -j KUBE-SEP-PATXOTJBHFPU4CNS
COMMIT
# Completed on Wed Oct 26 02:29:33 2016`),
			result: map[Chain][]Rule{
				ChainPrerouting: {
					{
						Modules: []string{"comment"},
						Options: map[string]RuleOption{
							"--comment": {Arg: "kubernetes service portals"},
							"-j":        {Arg: "KUBE-SERVICES"},
						},
					},
					{
						Modules: []string{"addrtype"},
						Options: map[string]RuleOption{
							"--dst-type": {Arg: "LOCAL"},
							"-j":         {Arg: "DOCKER"},
						},
					},
					{
						Modules: []string{"addrtype", "comment"},
						Options: map[string]RuleOption{
							"--dst-type": {Arg: "LOCAL"},
							"--comment":  {Arg: "handle service NodePorts; NOTE: this must be the last rule in the chain"},
							"-j":         {Arg: "KUBE-NODEPORT-CONTAINER"},
						},
					},
				},
				Chain("INPUT"): {},
				Chain("OUTPUT"): {
					{
						Modules: []string{"comment"},
						Options: map[string]RuleOption{
							"--comment": {Arg: "kubernetes service portals"},
							"-j":        {Arg: "KUBE-SERVICES"},
						},
					},
					{
						Modules: []string{"comment"},
						Options: map[string]RuleOption{
							"--comment": {Arg: "handle ClusterIPs; NOTE: this must be before the NodePort rules"},
							"-j":        {Arg: "KUBE-PORTALS-HOST"},
						},
					},
					{
						Modules: []string{"addrtype"},
						Options: map[string]RuleOption{
							"--dst-type": {Arg: "LOCAL"},
							"-d":         {Arg: "127.0.0.0/8", Negated: true},
							"-j":         {Arg: "DOCKER"},
						},
					},
					{
						Modules: []string{"addrtype", "comment"},
						Options: map[string]RuleOption{
							"--dst-type": {Arg: "LOCAL"},
							"--comment":  {Arg: "handle service NodePorts; NOTE: this must be the last rule in the chain"},
							"-j":         {Arg: "KUBE-NODEPORT-HOST"},
						},
					},
				},
				Chain("POSTROUTING"): {
					{
						Modules: []string{"comment"},
						Options: map[string]RuleOption{
							"--comment": {Arg: "kubernetes postrouting rules"},
							"-j":        {Arg: "KUBE-POSTROUTING"},
						},
					},
					{
						Options: map[string]RuleOption{
							"-s": {Arg: "10.128.0.0/14"},
							"-j": {Arg: "MASQUERADE"},
						},
					},
					{
						Options: map[string]RuleOption{
							"-s": {Arg: "172.18.0.0/16"},
							"-o": {Arg: "docker0", Negated: true},
							"-j": {Arg: "MASQUERADE"},
						},
					},
				},
				Chain("DOCKER"): {
					{
						Options: map[string]RuleOption{
							"-i": {Arg: "docker0"},
							"-j": {Arg: "RETURN"},
						},
					},
				},
				Chain("KUBE-MARK-DROP"): {
					{
						Options: map[string]RuleOption{
							"--set-xmark": {Arg: "0x8000/0x8000"},
							"-j":          {Arg: "MARK"},
						},
					},
				},
				Chain("KUBE-MARK-MASQ"): {
					{
						Options: map[string]RuleOption{
							"--set-xmark": {Arg: "0x4000/0x4000"},
							"-j":          {Arg: "MARK"},
						},
					},
				},
				Chain("KUBE-POSTROUTING"): {
					{
						Modules: []string{"comment", "mark"},
						Options: map[string]RuleOption{
							"--mark":    {Arg: "0x4000/0x4000"},
							"--comment": {Arg: "kubernetes service traffic requiring SNAT"},
							"-j":        {Arg: "MASQUERADE"},
						},
					},
				},
				Chain("KUBE-SEP-CKTKXEMIKRIIY55M"): {
					{
						Modules: []string{"comment"},
						Options: map[string]RuleOption{
							"-s":        {Arg: "172.17.0.2/32"},
							"--comment": {Arg: "default/kubernetes:dns"},
							"-j":        {Arg: "KUBE-MARK-MASQ"},
						},
					},
					{
						Modules: []string{"comment", "recent", "udp"},
						Options: map[string]RuleOption{
							"-p":               {Arg: "udp"},
							"--set":            {},
							"--name":           {Arg: "KUBE-SEP-CKTKXEMIKRIIY55M"},
							"--mask":           {Arg: "255.255.255.255"},
							"--to-destination": {Arg: "172.17.0.2:8053"},
							"--rsource":        {},
							"--comment":        {Arg: "default/kubernetes:dns"},
							"-j":               {Arg: "DNAT"},
						},
					},
				},
				Chain("KUBE-SEP-EZ5ESXJRZ36JV4D4"): {
					{
						Modules: []string{"comment"},
						Options: map[string]RuleOption{
							"-s":        {Arg: "172.17.0.2/32"},
							"--comment": {Arg: "default/kubernetes:dns-tcp"},
							"-j":        {Arg: "KUBE-MARK-MASQ"},
						},
					},
					{
						Modules: []string{"comment", "recent", "tcp"},
						Options: map[string]RuleOption{
							"-p":               {Arg: "tcp"},
							"--set":            {},
							"--name":           {Arg: "KUBE-SEP-EZ5ESXJRZ36JV4D4"},
							"--mask":           {Arg: "255.255.255.255"},
							"--to-destination": {Arg: "172.17.0.2:8053"},
							"--rsource":        {},
							"--comment":        {Arg: "default/kubernetes:dns-tcp"},
							"-j":               {Arg: "DNAT"},
						},
					},
				},
				Chain("KUBE-SEP-PATXOTJBHFPU4CNS"): {
					{
						Modules: []string{"comment"},
						Options: map[string]RuleOption{
							"-s":        {Arg: "172.17.0.2/32"},
							"--comment": {Arg: "default/kubernetes:https"},
							"-j":        {Arg: "KUBE-MARK-MASQ"},
						},
					},
					{
						Modules: []string{"comment", "recent", "tcp"},
						Options: map[string]RuleOption{
							"-p":               {Arg: "tcp"},
							"--set":            {},
							"--name":           {Arg: "KUBE-SEP-PATXOTJBHFPU4CNS"},
							"--mask":           {Arg: "255.255.255.255"},
							"--to-destination": {Arg: "172.17.0.2:8443"},
							"--rsource":        {},
							"--comment":        {Arg: "default/kubernetes:https"},
							"-j":               {Arg: "DNAT"},
						},
					},
				},
				Chain("KUBE-SERVICES"): {
					{
						Modules: []string{"comment", "tcp"},
						Options: map[string]RuleOption{
							"-d":        {Arg: "172.30.0.1/32"},
							"-p":        {Arg: "tcp"},
							"--dport":   {Arg: "443"},
							"--comment": {Arg: "default/kubernetes:https cluster IP"},
							"-j":        {Arg: "KUBE-SVC-NPX46M4PTMTKRN6Y"},
						},
					},
					{
						Modules: []string{"comment", "udp"},
						Options: map[string]RuleOption{
							"-d":        {Arg: "172.30.0.1/32"},
							"-p":        {Arg: "udp"},
							"--dport":   {Arg: "53"},
							"--comment": {Arg: "default/kubernetes:dns cluster IP"},
							"-j":        {Arg: "KUBE-SVC-3VQ6B3MLH7E2SZT4"},
						},
					},
					{
						Modules: []string{"comment", "tcp"},
						Options: map[string]RuleOption{
							"-d":        {Arg: "172.30.0.1/32"},
							"-p":        {Arg: "tcp"},
							"--dport":   {Arg: "53"},
							"--comment": {Arg: "default/kubernetes:dns-tcp cluster IP"},
							"-j":        {Arg: "KUBE-SVC-BA6I5HTZKAAAJT56"},
						},
					},
					{
						Modules: []string{"addrtype", "comment", "physdev", "tcp"},
						Options: map[string]RuleOption{
							"-d":              {Arg: "172.46.148.244/32"},
							"-p":              {Arg: "tcp"},
							"--dport":         {Arg: "82"},
							"--physdev-is-in": {Negated: true},
							"--src-type":      {Arg: "LOCAL", Negated: true},
							"--comment":       {Arg: "default/nginxservice: external IP"},
							"-j":              {Arg: "KUBE-SVC-URRHIARQWDHNXJTW"},
						},
					},
					{
						Modules: []string{"addrtype", "comment", "tcp"},
						Options: map[string]RuleOption{
							"-d":         {Arg: "172.46.148.244/32"},
							"-p":         {Arg: "tcp"},
							"--dport":    {Arg: "82"},
							"--dst-type": {Arg: "LOCAL"},
							"--comment":  {Arg: "default/nginxservice: external IP"},
							"-j":         {Arg: "KUBE-SVC-URRHIARQWDHNXJTW"},
						},
					},
					{
						Modules: []string{"addrtype", "comment"},
						Options: map[string]RuleOption{
							"--dst-type": {Arg: "LOCAL"},
							"--comment":  {Arg: "kubernetes service nodeports; NOTE: this must be the last rule in this chain"},
							"-j":         {Arg: "KUBE-NODEPORTS"},
						},
					},
				},
				Chain("KUBE-SVC-3VQ6B3MLH7E2SZT4"): {
					{
						Modules: []string{"comment", "recent"},
						Options: map[string]RuleOption{
							"--rcheck":  {},
							"--seconds": {Arg: "180"},
							"--reap":    {},
							"--name":    {Arg: "KUBE-SEP-CKTKXEMIKRIIY55M"},
							"--mask":    {Arg: "255.255.255.255"},
							"--rsource": {},
							"--comment": {Arg: "default/kubernetes:dns"},
							"-j":        {Arg: "KUBE-SEP-CKTKXEMIKRIIY55M"},
						},
					},
					{
						Modules: []string{"comment"},
						Options: map[string]RuleOption{
							"--comment": {Arg: "default/kubernetes:dns"},
							"-j":        {Arg: "KUBE-SEP-CKTKXEMIKRIIY55M"},
						},
					},
				},
				Chain("KUBE-SVC-BA6I5HTZKAAAJT56"): {
					{
						Modules: []string{"comment", "recent"},
						Options: map[string]RuleOption{
							"--rcheck":  {},
							"--seconds": {Arg: "180"},
							"--reap":    {},
							"--name":    {Arg: "KUBE-SEP-EZ5ESXJRZ36JV4D4"},
							"--mask":    {Arg: "255.255.255.255"},
							"--rsource": {},
							"--comment": {Arg: "default/kubernetes:dns-tcp"},
							"-j":        {Arg: "KUBE-SEP-EZ5ESXJRZ36JV4D4"},
						},
					},
					{
						Modules: []string{"comment"},
						Options: map[string]RuleOption{
							"--comment": {Arg: "default/kubernetes:dns-tcp"},
							"-j":        {Arg: "KUBE-SEP-EZ5ESXJRZ36JV4D4"},
						},
					},
				},
				Chain("KUBE-SVC-NPX46M4PTMTKRN6Y"): {
					{
						Modules: []string{"comment", "recent"},
						Options: map[string]RuleOption{
							"--rcheck":  {},
							"--seconds": {Arg: "180"},
							"--reap":    {},
							"--name":    {Arg: "KUBE-SEP-PATXOTJBHFPU4CNS"},
							"--mask":    {Arg: "255.255.255.255"},
							"--rsource": {},
							"--comment": {Arg: "default/kubernetes:https"},
							"-j":        {Arg: "KUBE-SEP-PATXOTJBHFPU4CNS"},
						},
					},
					{
						Modules: []string{"comment"},
						Options: map[string]RuleOption{
							"--comment": {Arg: "default/kubernetes:https"},
							"-j":        {Arg: "KUBE-SEP-PATXOTJBHFPU4CNS"},
						},
					},
				},
			},
		},
		{
			table: TableNAT,
			data: []byte(`*nat
:KUBE-SVC-NPX46M4PTMTKRN6Y - [0:0]
-A OUTPUT -m comment --comment "blah blabh" -j KUBE-PORTALS-HOST
COMMIT`),
			result: map[Chain][]Rule{},
			err:    "Chain OUTPUT unknown",
		},
		{
			table: TableNAT,
			data: []byte(`*nat
:OUTPUT - [0:0]
-A OUTPUT -m comment --comment "blah blah -j KUBE-PORTALS-HOST
COMMIT`),
			result: map[Chain][]Rule{},
			err:    "invalid iptables rule (unterminated quote)",
		},
		{
			table: TableNAT,
			data: []byte(`*nat
:OUTPUT - [0:0]
-A OUTPUT -A FOOBAR -j KUBE-PORTALS-HOST
COMMIT`),
			result: map[Chain][]Rule{},
			err:    "invalid iptables rule (chain already specified)",
		},
		{
			table: TableNAT,
			data: []byte(`*nat
:OUTPUT - [0:0]
-A OUTPUT -d foobar -d blahblah -j KUBE-PORTALS-HOST
COMMIT`),
			result: map[Chain][]Rule{},
			err:    "invalid iptables rule (multiple \"-d\" args)",
		},
		{
			table: TableNAT,
			data: []byte(`*nat
:OUTPUT - [0:0]
-A OUTPUT -j KUBE-PORTALS-HOST -j KUBE-SOMETHING
COMMIT`),
			result: map[Chain][]Rule{},
			err:    "invalid iptables rule (multiple \"-j\" args)",
		},
		{
			table: TableNAT,
			data: []byte(`*nat
:OUTPUT - [0:0]
-A OUTPUT j
COMMIT`),
			result: map[Chain][]Rule{},
			err:    "invalid iptables rule (unknown option \"j\")",
		},
		{
			table: TableNAT,
			data: []byte(`*nat
:OUTPUT - [0:0]
-A OUTPUT -d SDFSFSDF !
COMMIT`),
			result: map[Chain][]Rule{},
			err:    "invalid iptables rule (not enough arguments)",
		},
		{
			table: TableNAT,
			data: []byte(`*nat
:PREROUTING ACCEPT [0:0]
:POSTROUTING ACCEPT [0:0]
:KUBE-POSTROUTING - [0:0]
:KUBE-SEP-CKTKXEMIKRIIY55M - [0:0]
:KUBE-SVC-BA6I5HTZKAAAJT56 - [0:0]
:KUBE-SVC-NPX46M4PTMTKRN6Y - [0:0]
-A PREROUTING -j KUBE-SERVICES
-A POSTROUTING -s 172.18.0.0/16 -j MASQUERADE
-A KUBE-POSTROUTING -j MASQUERADE
-A KUBE-SEP-CKTKXEMIKRIIY55M -j DNAT
-A KUBE-SVC-BA6I5HTZKAAAJT56 -j KUBE-SEP-EZ5ESXJRZ36JV4D4
-A KUBE-SVC-NPX46M4PTMTKRN6Y -j KUBE-SEP-PATXOTJBHFPU4CNS
COMMIT`),
			filterChains:        []Chain{ChainPrerouting, Chain("KUBE-POSTROUTING")},
			filterChainPrefixes: []string{"KUBE-SEP-", "KUBE-SVC-"},
			result: map[Chain][]Rule{
				ChainPrerouting: {
					{
						Options: map[string]RuleOption{
							"-j": {Arg: "KUBE-SERVICES"},
						},
					},
				},
				Chain("KUBE-POSTROUTING"): {
					{
						Options: map[string]RuleOption{
							"-j": {Arg: "MASQUERADE"},
						},
					},
				},
				Chain("KUBE-SEP-CKTKXEMIKRIIY55M"): {
					{
						Options: map[string]RuleOption{
							"-j": {Arg: "DNAT"},
						},
					},
				},
				Chain("KUBE-SVC-BA6I5HTZKAAAJT56"): {
					{
						Options: map[string]RuleOption{
							"-j": {Arg: "KUBE-SEP-EZ5ESXJRZ36JV4D4"},
						},
					},
				},
				Chain("KUBE-SVC-NPX46M4PTMTKRN6Y"): {
					{
						Options: map[string]RuleOption{
							"-j": {Arg: "KUBE-SEP-PATXOTJBHFPU4CNS"},
						},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		result, err := ParseTableAddRules(tc.table, tc.filterChains, tc.filterChainPrefixes, tc.data)
		if err != nil {
			if tc.err != "" {
				if !strings.HasPrefix(err.Error(), tc.err) {
					t.Errorf("expected error %v, got %v", tc.err, err)
				}
			} else {
				t.Errorf("unexpected error %v", err)
			}
		} else {
			if tc.err != "" {
				t.Errorf("expected error %v", tc.err)
			}

			// Explode equality checks so its easier to figure out
			// what rule is bad when something is wrong
			if len(result) != len(tc.result) {
				t.Errorf("got number of chains %v, expected %v", len(result), len(tc.result))
			}
			for k, v := range result {
				v2, ok := tc.result[k]
				if !ok {
					t.Errorf("chain %v not found in testcase result", k)
				}

				if len(v) != len(v2) {
					t.Errorf("chain %v number of rules %v not expected number of rules %v", k, len(v), len(v2))
				}

				for i, r := range v {
					if !reflect.DeepEqual(r, v2[i]) {
						t.Errorf("result chain %v rule %#v not equal to expected %#v", k, r, v2[i])
					}
				}

			}
		}
	}
}
