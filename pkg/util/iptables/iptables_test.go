// +build linux

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
	"bytes"
	"net"
	"os"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/dbus"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

const TestLockfilePath = "xtables.lock"

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
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
			// Exists.
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 1} },
			// Failure.
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 2} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
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
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	cmd := getIPTablesCommand(protocol)
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll(cmd, "-t", "nat", "-N", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
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
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
			// Failure.
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	// Success.
	err := runner.FlushChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll("iptables", "-t", "nat", "-F", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
	// Failure.
	err = runner.FlushChain(TableNAT, Chain("FOOBAR"))
	if err == nil {
		t.Errorf("expected failure")
	}
}

func TestDeleteChain(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
			// Failure.
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	// Success.
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll("iptables", "-t", "nat", "-X", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
	// Failure.
	err = runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err == nil {
		t.Errorf("expected failure")
	}
}

func TestEnsureRuleAlreadyExists(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// iptables-restore version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Success of that exec means "done".
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
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
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll("iptables", "-t", "nat", "-C", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
}

func TestEnsureRuleNew(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			// Status 1 on the first call.
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 1} },
			// Success on the second call.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// iptables-restore version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
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
	if fcmd.CombinedOutputCalls != 4 {
		t.Errorf("expected 4 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[3]...).HasAll("iptables", "-t", "nat", "-A", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[3])
	}
}

func TestEnsureRuleErrorChecking(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			// Status 2 on the first call.
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 2} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// iptables-restore version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
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

func TestEnsureRuleErrorCreating(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			// Status 1 on the first call.
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 1} },
			// Status 1 on the second call.
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// iptables-restore version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	_, err := runner.EnsureRule(Append, TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 4 {
		t.Errorf("expected 4 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
}

func TestDeleteRuleDoesNotExist(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			// Status 1 on the first call.
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// iptables-restore version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Failure of that exec means "does not exist".
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
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
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll("iptables", "-t", "nat", "-C", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
}

func TestDeleteRuleExists(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			// Success on the first call.
			func() ([]byte, error) { return []byte{}, nil },
			// Success on the second call.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// iptables-restore version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Success of that means delete it.
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 4 {
		t.Errorf("expected 4 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[3]...).HasAll("iptables", "-t", "nat", "-D", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[3])
	}
}

func TestDeleteRuleErrorChecking(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			// Status 2 on the first call.
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 2} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// iptables-restore version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
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

func TestDeleteRuleErrorDeleting(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			// Success on the first call.
			func() ([]byte, error) { return []byte{}, nil },
			// Status 1 on the second call.
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// iptables-restore version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Success of that means delete it.
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 4 {
		t.Errorf("expected 4 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
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
		fcmd := fakeexec.FakeCmd{
			CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
				func() ([]byte, error) { return []byte(testCase.Version), nil },
			},
		}
		fexec := fakeexec.FakeExec{
			CommandScript: []fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
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

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// Success.
			func() ([]byte, error) { return []byte(iptables_save_output), nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// The first Command() call is checking the rule.  Success of that exec means "done".
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
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

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// Success.
			func() ([]byte, error) { return []byte(iptables_save_output), nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// The first Command() call is checking the rule.  Success of that exec means "done".
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
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
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.4.19"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// iptables-restore version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if sets.NewString(fcmd.CombinedOutputLog[2]...).HasAny("-w", "-w2") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
}

func TestWaitFlagOld(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.4.20"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll("iptables", "-w") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
	if sets.NewString(fcmd.CombinedOutputLog[2]...).HasAny("-w2") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
}

func TestWaitFlagNew(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.4.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll("iptables", "-w2") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
	if sets.NewString(fcmd.CombinedOutputLog[2]...).HasAny("-w") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
}

func TestReload(t *testing.T) {
	dbusConn := dbus.NewFakeConnection()
	dbusConn.SetBusObject(func(method string, args ...interface{}) ([]interface{}, error) { return nil, nil })
	dbusConn.AddObject(firewalldName, firewalldPath, func(method string, args ...interface{}) ([]interface{}, error) { return nil, nil })
	fdbus := dbus.NewFake(dbusConn, nil)

	reloaded := make(chan bool, 2)

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.4.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },

			// first reload
			// EnsureChain
			func() ([]byte, error) { return []byte{}, nil },
			// EnsureRule abc check
			func() ([]byte, error) { return []byte{}, &fakeexec.FakeExitError{Status: 1} },
			// EnsureRule abc
			func() ([]byte, error) { return []byte{}, nil },

			// second reload
			// EnsureChain
			func() ([]byte, error) { return []byte{}, nil },
			// EnsureRule abc check
			func() ([]byte, error) { return []byte{}, &fakeexec.FakeExitError{Status: 1} },
			// EnsureRule abc
			func() ([]byte, error) { return []byte{}, nil },
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
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
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

	if fcmd.CombinedOutputCalls != 5 {
		t.Errorf("expected 5 CombinedOutput() calls total, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll("iptables", "-t", "nat", "-N", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
	if !sets.NewString(fcmd.CombinedOutputLog[3]...).HasAll("iptables", "-t", "nat", "-C", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[3])
	}
	if !sets.NewString(fcmd.CombinedOutputLog[4]...).HasAll("iptables", "-t", "nat", "-A", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[4])
	}

	go func() { time.Sleep(time.Second / 100); reloaded <- true }()
	dbusConn.EmitSignal(firewalldName, firewalldPath, firewalldInterface, "DefaultZoneChanged", "public")
	dbusConn.EmitSignal("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "NameOwnerChanged", "io.k8s.Something", "", ":1.1")
	<-reloaded

	if fcmd.CombinedOutputCalls != 5 {
		t.Errorf("Incorrect signal caused a reload")
	}

	dbusConn.EmitSignal(firewalldName, firewalldPath, firewalldInterface, "Reloaded")
	<-reloaded
	<-reloaded

	if fcmd.CombinedOutputCalls != 8 {
		t.Errorf("expected 8 CombinedOutput() calls total, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[5]...).HasAll("iptables", "-t", "nat", "-N", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[5])
	}
	if !sets.NewString(fcmd.CombinedOutputLog[6]...).HasAll("iptables", "-t", "nat", "-C", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[6])
	}
	if !sets.NewString(fcmd.CombinedOutputLog[7]...).HasAll("iptables", "-t", "nat", "-A", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[7])
	}
}

func TestSaveInto(t *testing.T) {
	output := `# Generated by iptables-save v1.6.0 on Thu Jan 19 11:38:09 2017
*filter
:INPUT ACCEPT [15079:38410730]
:FORWARD ACCEPT [0:0]
:OUTPUT ACCEPT [11045:521562]
COMMIT
# Completed on Thu Jan 19 11:38:09 2017`

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
		},
		RunScript: []fakeexec.FakeRunAction{
			func() ([]byte, []byte, error) { return []byte(output), nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	buffer := bytes.NewBuffer(nil)

	// Success.
	err := runner.SaveInto(TableNAT, buffer)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	if string(buffer.Bytes()[:len(output)]) != output {
		t.Errorf("expected output to be equal to mocked one, got %v", buffer.Bytes())
	}

	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if fcmd.RunCalls != 1 {
		t.Errorf("expected 1 Run() call, got %d", fcmd.RunCalls)
	}
	if !sets.NewString(fcmd.RunLog[0]...).HasAll("iptables-save", "-t", "nat") {
		t.Errorf("wrong Run() log, got %s", fcmd.RunLog[0])
	}

	// Failure.
	buffer.Reset()
	err = runner.SaveInto(TableNAT, buffer)
	if err == nil {
		t.Errorf("expected failure")
	}
}

func TestRestore(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			func() ([]byte, error) { return []byte{}, nil },
			func() ([]byte, error) { return []byte{}, nil },
			func() ([]byte, error) { return []byte{}, nil },
			func() ([]byte, error) { return []byte{}, nil },
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 1} },
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
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()

	// both flags true
	err := runner.Restore(TableNAT, []byte{}, FlushTables, RestoreCounters)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}

	commandSet := sets.NewString(fcmd.CombinedOutputLog[2]...)
	if !commandSet.HasAll("iptables-restore", "-T", string(TableNAT), "--counters") || commandSet.HasAny("--noflush") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}

	// FlushTables, NoRestoreCounters
	err = runner.Restore(TableNAT, []byte{}, FlushTables, NoRestoreCounters)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}

	commandSet = sets.NewString(fcmd.CombinedOutputLog[3]...)
	if !commandSet.HasAll("iptables-restore", "-T", string(TableNAT)) || commandSet.HasAny("--noflush", "--counters") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}

	// NoFlushTables, RestoreCounters
	err = runner.Restore(TableNAT, []byte{}, NoFlushTables, RestoreCounters)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}

	commandSet = sets.NewString(fcmd.CombinedOutputLog[4]...)
	if !commandSet.HasAll("iptables-restore", "-T", string(TableNAT), "--noflush", "--counters") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[3])
	}

	// NoFlushTables, NoRestoreCounters
	err = runner.Restore(TableNAT, []byte{}, NoFlushTables, NoRestoreCounters)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}

	commandSet = sets.NewString(fcmd.CombinedOutputLog[5]...)
	if !commandSet.HasAll("iptables-restore", "-T", string(TableNAT), "--noflush") || commandSet.HasAny("--counters") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[4])
	}

	if fcmd.CombinedOutputCalls != 6 {
		t.Errorf("expected 6 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}

	// Failure.
	err = runner.Restore(TableNAT, []byte{}, FlushTables, RestoreCounters)
	if err == nil {
		t.Errorf("expected failure")
	}
}

// TestRestoreAll tests only the simplest use case, as flag handling code is already tested in TestRestore
func TestRestoreAll(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			func() ([]byte, error) { return []byte{}, nil },
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := newInternal(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4, TestLockfilePath)
	defer os.Remove(TestLockfilePath)
	defer runner.Destroy()

	err := runner.RestoreAll([]byte{}, NoFlushTables, RestoreCounters)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	commandSet := sets.NewString(fcmd.CombinedOutputLog[2]...)
	if !commandSet.HasAll("iptables-restore", "--counters", "--noflush") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}

	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}

	// Failure.
	err = runner.Restore(TableNAT, []byte{}, FlushTables, RestoreCounters)
	if err == nil {
		t.Errorf("expected failure")
	}
}

// TestRestoreAllWait tests that the "wait" flag is passed to a compatible iptables-restore
func TestRestoreAllWait(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("iptables-restore v1.9.22"), nil },
			func() ([]byte, error) { return []byte{}, nil },
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := newInternal(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4, TestLockfilePath)
	defer os.Remove(TestLockfilePath)
	defer runner.Destroy()

	err := runner.RestoreAll([]byte{}, NoFlushTables, RestoreCounters)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	commandSet := sets.NewString(fcmd.CombinedOutputLog[2]...)
	if !commandSet.HasAll("iptables-restore", "--wait=2", "--counters", "--noflush") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}

	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}

	// Failure.
	err = runner.Restore(TableNAT, []byte{}, FlushTables, RestoreCounters)
	if err == nil {
		t.Errorf("expected failure")
	}
}

// TestRestoreAllWaitOldIptablesRestore tests that the "wait" flag is not passed
// to a in-compatible iptables-restore
func TestRestoreAllWaitOldIptablesRestore(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("unrecognized option: --version"), nil },
			func() ([]byte, error) { return []byte{}, nil },
			func() ([]byte, error) { return nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := newInternal(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4, TestLockfilePath)
	defer os.Remove(TestLockfilePath)
	defer runner.Destroy()

	err := runner.RestoreAll([]byte{}, NoFlushTables, RestoreCounters)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	commandSet := sets.NewString(fcmd.CombinedOutputLog[2]...)
	if !commandSet.HasAll("iptables-restore", "--counters", "--noflush") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
	if commandSet.HasAny("--wait=2") {
		t.Errorf("wrong CombinedOutput() log (unexpected --wait=2 option), got %s", fcmd.CombinedOutputLog[2])
	}

	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}

	// Failure.
	err = runner.Restore(TableNAT, []byte{}, FlushTables, RestoreCounters)
	if err == nil {
		t.Errorf("expected failure")
	}
}

// TestRestoreAllGrabNewLock tests that the iptables code will grab the
// iptables /run lock when using an iptables-restore version that does not
// support the --wait argument
func TestRestoreAllGrabNewLock(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("unrecognized option: --version"), nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}

	runner := newInternal(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4, TestLockfilePath)
	defer os.Remove(TestLockfilePath)
	defer runner.Destroy()

	// Grab the /run lock and ensure the RestoreAll fails
	runLock, err := os.OpenFile(TestLockfilePath, os.O_CREATE, 0600)
	if err != nil {
		t.Fatalf("expected to open %s, got %v", TestLockfilePath, err)
	}
	defer runLock.Close()

	if err := grabIptablesFileLock(runLock); err != nil {
		t.Errorf("expected to lock %s, got %v", TestLockfilePath, err)
	}

	err = runner.RestoreAll([]byte{}, NoFlushTables, RestoreCounters)
	if err == nil {
		t.Errorf("expected failure, got success instead")
	}
	if !strings.Contains(err.Error(), "failed to acquire new iptables lock: timed out waiting for the condition") {
		t.Errorf("expected timeout error, got %v", err)
	}
}

// TestRestoreAllGrabOldLock tests that the iptables code will grab the
// iptables @xtables abstract unix socket lock when using an iptables-restore
// version that does not support the --wait argument
func TestRestoreAllGrabOldLock(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// iptables-restore version check
			func() ([]byte, error) { return []byte("unrecognized option: --version"), nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}

	runner := newInternal(&fexec, dbus.NewFake(nil, nil), ProtocolIpv4, TestLockfilePath)
	defer os.Remove(TestLockfilePath)
	defer runner.Destroy()

	// Grab the abstract @xtables socket
	runLock, err := net.ListenUnix("unix", &net.UnixAddr{Name: "@xtables", Net: "unix"})
	if err != nil {
		t.Fatalf("expected to lock @xtables, got %v", err)
	}
	defer runLock.Close()

	err = runner.RestoreAll([]byte{}, NoFlushTables, RestoreCounters)
	if err == nil {
		t.Errorf("expected failure, got success instead")
	}
	if !strings.Contains(err.Error(), "failed to acquire old iptables lock: timed out waiting for the condition") {
		t.Errorf("expected timeout error, got %v", err)
	}
}
