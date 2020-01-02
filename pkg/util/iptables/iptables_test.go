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
	"fmt"
	"net"
	"os"
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

const TestLockfilePath = "xtables.lock"

func protocolStr(protocol Protocol) string {
	if protocol == ProtocolIpv4 {
		return "IPv4"
	}
	return "IPv6"
}

func testIPTablesVersionCmds(t *testing.T, protocol Protocol) {
	version := " v1.4.22"
	iptablesCmd := iptablesCommand(protocol)
	iptablesRestoreCmd := iptablesRestoreCommand(protocol)
	protoStr := protocolStr(protocol)

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version response (for runner instantiation)
			func() ([]byte, []byte, error) { return []byte(iptablesCmd + version), nil, nil },
			// iptables-restore version response (for runner instantiation)
			func() ([]byte, []byte, error) { return []byte(iptablesRestoreCmd + version), nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	_ = New(&fexec, protocol)

	// Check that proper iptables version command was used during runner instantiation
	if !sets.NewString(fcmd.CombinedOutputLog[0]...).HasAll(iptablesCmd, "--version") {
		t.Errorf("%s runner instantiate: Expected cmd '%s --version', Got '%s'", protoStr, iptablesCmd, fcmd.CombinedOutputLog[0])
	}

	// Check that proper iptables restore version command was used during runner instantiation
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll(iptablesRestoreCmd, "--version") {
		t.Errorf("%s runner instantiate: Expected cmd '%s --version', Got '%s'", protoStr, iptablesRestoreCmd, fcmd.CombinedOutputLog[1])
	}
}

func TestIPTablesVersionCmdsIPv4(t *testing.T) {
	testIPTablesVersionCmds(t, ProtocolIpv4)
}

func TestIPTablesVersionCmdsIPv6(t *testing.T) {
	testIPTablesVersionCmds(t, ProtocolIpv6)
}

func testEnsureChain(t *testing.T, protocol Protocol) {
	protoStr := protocolStr(protocol)

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.9.22"), nil, nil },
			// Success.
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			// Exists.
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
			// Failure.
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 2} },
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
	runner := New(&fexec, protocol)
	// Success.
	exists, err := runner.EnsureChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("%s new chain: Expected success, got %v", protoStr, err)
	}
	if exists {
		t.Errorf("%s new chain: Expected exists = false", protoStr)
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("%s new chain: Expected 2 CombinedOutput() calls, got %d", protoStr, fcmd.CombinedOutputCalls)
	}
	cmd := iptablesCommand(protocol)
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll(cmd, "-t", "nat", "-N", "FOOBAR") {
		t.Errorf("%s new chain: Expected cmd containing '%s -t nat -N FOOBAR', got %s", protoStr, cmd, fcmd.CombinedOutputLog[2])
	}
	// Exists.
	exists, err = runner.EnsureChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("%s existing chain: Expected success, got %v", protoStr, err)
	}
	if !exists {
		t.Errorf("%s existing chain: Expected exists = true", protoStr)
	}
	// Simulate failure.
	_, err = runner.EnsureChain(TableNAT, Chain("FOOBAR"))
	if err == nil {
		t.Errorf("%s: Expected failure", protoStr)
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
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.9.22"), nil, nil },
			// Success.
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			// Failure.
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, ProtocolIpv4)
	// Success.
	err := runner.FlushChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll("iptables", "-t", "nat", "-F", "FOOBAR") {
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
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.9.22"), nil, nil },
			// Success.
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			// Failure.
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, ProtocolIpv4)
	// Success.
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll("iptables", "-t", "nat", "-X", "FOOBAR") {
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
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.9.22"), nil, nil },
			// Success.
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Success of that exec means "done".
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, ProtocolIpv4)
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
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
}

func TestEnsureRuleNew(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.9.22"), nil, nil },
			// Status 1 on the first call.
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
			// Success on the second call.
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, ProtocolIpv4)
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
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[3])
	}
}

func TestEnsureRuleErrorChecking(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.9.22"), nil, nil },
			// Status 2 on the first call.
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 2} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, ProtocolIpv4)
	_, err := runner.EnsureRule(Append, TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
}

func TestEnsureRuleErrorCreating(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.9.22"), nil, nil },
			// Status 1 on the first call.
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
			// Status 1 on the second call.
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, ProtocolIpv4)
	_, err := runner.EnsureRule(Append, TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
}

func TestDeleteRuleDoesNotExist(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.9.22"), nil, nil },
			// Status 1 on the first call.
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Failure of that exec means "does not exist".
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, ProtocolIpv4)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll("iptables", "-t", "nat", "-C", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
}

func TestDeleteRuleExists(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.9.22"), nil, nil },
			// Success on the first call.
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			// Success on the second call.
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Success of that means delete it.
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, ProtocolIpv4)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll("iptables", "-t", "nat", "-D", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[3])
	}
}

func TestDeleteRuleErrorChecking(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.9.22"), nil, nil },
			// Status 2 on the first call.
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 2} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, ProtocolIpv4)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
}

func TestDeleteRuleErrorDeleting(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.9.22"), nil, nil },
			// Success on the first call.
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			// Status 1 on the second call.
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			// iptables version check
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			// The second Command() call is checking the rule.  Success of that means delete it.
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, ProtocolIpv4)
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
		Expected bool
	}{
		{"iptables v1.4.7", false},
		{"iptables v1.4.11", true},
		{"iptables v1.4.19.1", true},
		{"iptables v2.0.0", true},
		{"total junk", true},
	}

	for _, testCase := range testCases {
		fcmd := fakeexec.FakeCmd{
			CombinedOutputScript: []fakeexec.FakeAction{
				func() ([]byte, []byte, error) { return []byte(testCase.Version), nil, nil },
				func() ([]byte, []byte, error) { return []byte(testCase.Version), nil, nil },
			},
		}
		fexec := fakeexec.FakeExec{
			CommandScript: []fakeexec.FakeCommandAction{
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			},
		}
		ipt := New(&fexec, ProtocolIpv4)
		runner := ipt.(*runner)
		if testCase.Expected != runner.hasCheck {
			t.Errorf("Expected result: %v, Got result: %v", testCase.Expected, runner.hasCheck)
		}
	}
}

func TestIPTablesCommands(t *testing.T) {
	testCases := []struct {
		funcName    string
		protocol    Protocol
		expectedCmd string
	}{
		{"iptablesCommand", ProtocolIpv4, cmdIPTables},
		{"iptablesCommand", ProtocolIpv6, cmdIP6Tables},
		{"iptablesSaveCommand", ProtocolIpv4, cmdIPTablesSave},
		{"iptablesSaveCommand", ProtocolIpv6, cmdIP6TablesSave},
		{"iptablesRestoreCommand", ProtocolIpv4, cmdIPTablesRestore},
		{"iptablesRestoreCommand", ProtocolIpv6, cmdIP6TablesRestore},
	}
	for _, testCase := range testCases {
		var cmd string
		switch testCase.funcName {
		case "iptablesCommand":
			cmd = iptablesCommand(testCase.protocol)
		case "iptablesSaveCommand":
			cmd = iptablesSaveCommand(testCase.protocol)
		case "iptablesRestoreCommand":
			cmd = iptablesRestoreCommand(testCase.protocol)
		}
		if cmd != testCase.expectedCmd {
			t.Errorf("Function: %s, Expected result: %s, Actual result: %s", testCase.funcName, testCase.expectedCmd, cmd)
		}
	}
}

func TestCheckRuleWithoutCheckPresent(t *testing.T) {
	iptablesSaveOutput := `# Generated by iptables-save v1.4.7 on Wed Oct 29 14:56:01 2014
*nat
:PREROUTING ACCEPT [2136997:197881818]
:POSTROUTING ACCEPT [4284525:258542680]
:OUTPUT ACCEPT [5901660:357267963]
-A PREROUTING -m addrtype --dst-type LOCAL -m mark --mark 0x00004000/0x00004000 -j DOCKER
COMMIT
# Completed on Wed Oct 29 14:56:01 2014`

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// Success.
			func() ([]byte, []byte, error) { return []byte(iptablesSaveOutput), nil, nil },
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
	iptablesSaveOutput := `# Generated by iptables-save v1.4.7 on Wed Oct 29 14:56:01 2014
*nat
:PREROUTING ACCEPT [2136997:197881818]
:POSTROUTING ACCEPT [4284525:258542680]
:OUTPUT ACCEPT [5901660:357267963]
-A PREROUTING -m addrtype --dst-type LOCAL -j DOCKER
COMMIT
# Completed on Wed Oct 29 14:56:01 2014`

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// Success.
			func() ([]byte, []byte, error) { return []byte(iptablesSaveOutput), nil, nil },
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
		Result  []string
	}{
		{"0.55.55", nil},
		{"1.0.55", nil},
		{"1.4.19", nil},
		{"1.4.20", []string{WaitString}},
		{"1.4.21", []string{WaitString}},
		{"1.4.22", []string{WaitString, WaitSecondsValue}},
		{"1.5.0", []string{WaitString, WaitSecondsValue}},
		{"2.0.0", []string{WaitString, WaitSecondsValue, WaitIntervalString, WaitIntervalUsecondsValue}},
	}

	for _, testCase := range testCases {
		result := getIPTablesWaitFlag(utilversion.MustParseGeneric(testCase.Version))
		if !reflect.DeepEqual(result, testCase.Result) {
			t.Errorf("For %s expected %v got %v", testCase.Version, testCase.Result, result)
		}
	}
}

func TestWaitFlagUnavailable(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.4.19"), nil, nil },
			// iptables-restore version check
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			// Success.
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
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
	runner := New(&fexec, ProtocolIpv4)
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if sets.NewString(fcmd.CombinedOutputLog[2]...).Has(WaitString) {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
}

func TestWaitFlagOld(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.4.20"), nil, nil },
			// iptables-restore version check
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			// Success.
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, ProtocolIpv4)
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll("iptables", WaitString) {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
	if sets.NewString(fcmd.CombinedOutputLog[2]...).Has(WaitSecondsValue) {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
}

func TestWaitFlagNew(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.4.22"), nil, nil },
			// iptables-restore version check
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			// Success.
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, ProtocolIpv4)
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll("iptables", WaitString, WaitSecondsValue) {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
}

func TestWaitIntervalFlagNew(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.6.1"), nil, nil },
			// iptables-restore version check
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			// Success.
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, ProtocolIpv4)
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll("iptables", WaitString, WaitSecondsValue, WaitIntervalString, WaitIntervalUsecondsValue) {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
}

func testSaveInto(t *testing.T, protocol Protocol) {
	version := " v1.9.22"
	iptablesCmd := iptablesCommand(protocol)
	iptablesSaveCmd := iptablesSaveCommand(protocol)
	protoStr := protocolStr(protocol)

	output := fmt.Sprintf(`# Generated by %s on Thu Jan 19 11:38:09 2017
*filter
:INPUT ACCEPT [15079:38410730]
:FORWARD ACCEPT [0:0]
:OUTPUT ACCEPT [11045:521562]
COMMIT
# Completed on Thu Jan 19 11:38:09 2017`, iptablesSaveCmd+version)

	stderrOutput := "#STDERR OUTPUT" // SaveInto() should should NOT capture stderr into the buffer

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte(iptablesCmd + version), nil, nil },
		},
		RunScript: []fakeexec.FakeAction{
			func() ([]byte, []byte, error) { return []byte(output), []byte(stderrOutput), nil },
			func() ([]byte, []byte, error) { return nil, []byte(stderrOutput), &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, protocol)
	buffer := bytes.NewBuffer(nil)

	// Success.
	err := runner.SaveInto(TableNAT, buffer)
	if err != nil {
		t.Fatalf("%s: Expected success, got %v", protoStr, err)
	}

	if string(buffer.Bytes()) != output {
		t.Errorf("%s: Expected output '%s', got '%v'", protoStr, output, string(buffer.Bytes()))
	}

	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("%s: Expected 1 CombinedOutput() calls, got %d", protoStr, fcmd.CombinedOutputCalls)
	}
	if fcmd.RunCalls != 1 {
		t.Errorf("%s: Expected 1 Run() call, got %d", protoStr, fcmd.RunCalls)
	}
	if !sets.NewString(fcmd.RunLog[0]...).HasAll(iptablesSaveCmd, "-t", "nat") {
		t.Errorf("%s: Expected cmd containing '%s -t nat', got '%s'", protoStr, iptablesSaveCmd, fcmd.RunLog[0])
	}

	// Failure.
	buffer.Reset()
	err = runner.SaveInto(TableNAT, buffer)
	if err == nil {
		t.Errorf("%s: Expected failure", protoStr)
	}
	if string(buffer.Bytes()) != stderrOutput {
		t.Errorf("%s: Expected output '%s', got '%v'", protoStr, stderrOutput, string(buffer.Bytes()))
	}
}

func TestSaveIntoIPv4(t *testing.T) {
	testSaveInto(t, ProtocolIpv4)
}

func TestSaveIntoIPv6(t *testing.T) {
	testSaveInto(t, ProtocolIpv6)
}

func testRestore(t *testing.T, protocol Protocol) {
	version := " v1.9.22"
	iptablesCmd := iptablesCommand(protocol)
	iptablesRestoreCmd := iptablesRestoreCommand(protocol)
	protoStr := protocolStr(protocol)

	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte(iptablesCmd + version), nil, nil },
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
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
	runner := New(&fexec, protocol)

	// both flags true
	err := runner.Restore(TableNAT, []byte{}, FlushTables, RestoreCounters)
	if err != nil {
		t.Errorf("%s flush,restore: Expected success, got %v", protoStr, err)
	}

	commandSet := sets.NewString(fcmd.CombinedOutputLog[1]...)
	if !commandSet.HasAll(iptablesRestoreCmd, "-T", string(TableNAT), "--counters") || commandSet.HasAny("--noflush") {
		t.Errorf("%s flush, restore: Expected cmd containing '%s -T %s --counters', got '%s'", protoStr, iptablesRestoreCmd, string(TableNAT), fcmd.CombinedOutputLog[1])
	}

	// FlushTables, NoRestoreCounters
	err = runner.Restore(TableNAT, []byte{}, FlushTables, NoRestoreCounters)
	if err != nil {
		t.Errorf("%s flush, no restore: Expected success, got %v", protoStr, err)
	}

	commandSet = sets.NewString(fcmd.CombinedOutputLog[2]...)
	if !commandSet.HasAll(iptablesRestoreCmd, "-T", string(TableNAT)) || commandSet.HasAny("--noflush", "--counters") {
		t.Errorf("%s flush, no restore: Expected cmd containing '--noflush' or '--counters', got '%s'", protoStr, fcmd.CombinedOutputLog[2])
	}

	// NoFlushTables, RestoreCounters
	err = runner.Restore(TableNAT, []byte{}, NoFlushTables, RestoreCounters)
	if err != nil {
		t.Errorf("%s no flush, restore: Expected success, got %v", protoStr, err)
	}

	commandSet = sets.NewString(fcmd.CombinedOutputLog[3]...)
	if !commandSet.HasAll(iptablesRestoreCmd, "-T", string(TableNAT), "--noflush", "--counters") {
		t.Errorf("%s no flush, restore: Expected cmd containing '--noflush' and '--counters', got '%s'", protoStr, fcmd.CombinedOutputLog[3])
	}

	// NoFlushTables, NoRestoreCounters
	err = runner.Restore(TableNAT, []byte{}, NoFlushTables, NoRestoreCounters)
	if err != nil {
		t.Errorf("%s no flush, no restore: Expected success, got %v", protoStr, err)
	}

	commandSet = sets.NewString(fcmd.CombinedOutputLog[4]...)
	if !commandSet.HasAll(iptablesRestoreCmd, "-T", string(TableNAT), "--noflush") || commandSet.HasAny("--counters") {
		t.Errorf("%s no flush, no restore: Expected cmd containing '%s -T %s --noflush', got '%s'", protoStr, iptablesRestoreCmd, string(TableNAT), fcmd.CombinedOutputLog[4])
	}

	if fcmd.CombinedOutputCalls != 5 {
		t.Errorf("%s: Expected 5 total CombinedOutput() calls, got %d", protoStr, fcmd.CombinedOutputCalls)
	}

	// Failure.
	err = runner.Restore(TableNAT, []byte{}, FlushTables, RestoreCounters)
	if err == nil {
		t.Errorf("%s Expected a failure", protoStr)
	}
}

func TestRestoreIPv4(t *testing.T) {
	testRestore(t, ProtocolIpv4)
}

func TestRestoreIPv6(t *testing.T) {
	testRestore(t, ProtocolIpv6)
}

// TestRestoreAll tests only the simplest use case, as flag handling code is already tested in TestRestore
func TestRestoreAll(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.9.22"), nil, nil },
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := newInternal(&fexec, ProtocolIpv4, TestLockfilePath)
	defer os.Remove(TestLockfilePath)

	err := runner.RestoreAll([]byte{}, NoFlushTables, RestoreCounters)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	commandSet := sets.NewString(fcmd.CombinedOutputLog[1]...)
	if !commandSet.HasAll("iptables-restore", "--counters", "--noflush") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}

	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
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
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.9.22"), nil, nil },
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			func() ([]byte, []byte, error) { return nil, nil, &fakeexec.FakeExitError{Status: 1} },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := newInternal(&fexec, ProtocolIpv4, TestLockfilePath)
	defer os.Remove(TestLockfilePath)

	err := runner.RestoreAll([]byte{}, NoFlushTables, RestoreCounters)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	commandSet := sets.NewString(fcmd.CombinedOutputLog[1]...)
	if !commandSet.HasAll("iptables-restore", WaitString, WaitSecondsValue, WaitIntervalString, WaitIntervalUsecondsValue, "--counters", "--noflush") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}

	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}

	// Failure.
	err = runner.Restore(TableNAT, []byte{}, FlushTables, RestoreCounters)
	if err == nil {
		t.Errorf("expected failure")
	}
}

// TestRestoreAllWaitOldIptablesRestore tests that the "wait" flag is not passed
// to an old iptables-restore
func TestRestoreAllWaitOldIptablesRestore(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.4.22"), nil, nil },
			// iptables-restore version check
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
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
	runner := newInternal(&fexec, ProtocolIpv4, TestLockfilePath)
	defer os.Remove(TestLockfilePath)

	err := runner.RestoreAll([]byte{}, NoFlushTables, RestoreCounters)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	commandSet := sets.NewString(fcmd.CombinedOutputLog[2]...)
	if !commandSet.HasAll("iptables-restore", "--counters", "--noflush") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
	if commandSet.HasAll(WaitString) {
		t.Errorf("wrong CombinedOutput() log (unexpected %s option), got %s", WaitString, fcmd.CombinedOutputLog[1])
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
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.4.22"), nil, nil },
			// iptables-restore version check
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}

	runner := newInternal(&fexec, ProtocolIpv4, TestLockfilePath)
	defer os.Remove(TestLockfilePath)

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
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.4.22"), nil, nil },
			// iptables-restore version check
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}

	runner := newInternal(&fexec, ProtocolIpv4, TestLockfilePath)
	defer os.Remove(TestLockfilePath)

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

// TestRestoreAllWaitBackportedIptablesRestore tests that the "wait" flag is passed
// to a seemingly-old-but-actually-new iptables-restore
func TestRestoreAllWaitBackportedIptablesRestore(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeAction{
			// iptables version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.4.22"), nil, nil },
			// iptables-restore version check
			func() ([]byte, []byte, error) { return []byte("iptables v1.4.22"), nil, nil },
			func() ([]byte, []byte, error) { return []byte{}, nil, nil },
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
	runner := newInternal(&fexec, ProtocolIpv4, TestLockfilePath)
	defer os.Remove(TestLockfilePath)

	err := runner.RestoreAll([]byte{}, NoFlushTables, RestoreCounters)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}

	commandSet := sets.NewString(fcmd.CombinedOutputLog[2]...)
	if !commandSet.HasAll("iptables-restore", "--counters", "--noflush") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[2])
	}
	if !commandSet.HasAll(WaitString) {
		t.Errorf("wrong CombinedOutput() log (expected %s option), got %s", WaitString, fcmd.CombinedOutputLog[1])
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
