/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"testing"

	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/exec"
)

func getIptablesCommand(protocol Protocol) string {
	if protocol == ProtocolIpv4 {
		return cmdIptables
	}
	if protocol == ProtocolIpv6 {
		return cmdIp6tables
	}
	panic("Unknown protocol")
}

func testEnsureChain(t *testing.T, protocol Protocol) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
			// Exists.
			func() ([]byte, error) { return nil, &exec.FakeExitError{1} },
			// Failure.
			func() ([]byte, error) { return nil, &exec.FakeExitError{2} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, protocol)
	// Success.
	exists, err := runner.EnsureChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if exists {
		t.Errorf("expected exists = false")
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.CombinedOutputCalls)
	}
	cmd := getIptablesCommand(protocol)
	if !util.NewStringSet(fcmd.CombinedOutputLog[0]...).HasAll(cmd, "-t", "nat", "-N", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
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
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
			// Failure.
			func() ([]byte, error) { return nil, &exec.FakeExitError{1} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, ProtocolIpv4)
	// Success.
	err := runner.FlushChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.CombinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.CombinedOutputLog[0]...).HasAll("iptables", "-t", "nat", "-F", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
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
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
			// Failure.
			func() ([]byte, error) { return nil, &exec.FakeExitError{1} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec, ProtocolIpv4)
	// Success.
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.CombinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.CombinedOutputLog[0]...).HasAll("iptables", "-t", "nat", "-X", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
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
	runner := New(&fexec, ProtocolIpv4)
	exists, err := runner.EnsureRule(Append, TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if !exists {
		t.Errorf("expected exists = true")
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() call, got %d", fcmd.CombinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.CombinedOutputLog[1]...).HasAll("iptables", "-t", "nat", "-C", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
	}
}

func TestEnsureRuleNew(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// Status 1 on the first call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{1} },
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
	if !util.NewStringSet(fcmd.CombinedOutputLog[2]...).HasAll("iptables", "-t", "nat", "-A", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
}

func TestEnsureRuleErrorChecking(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// Status 2 on the first call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{2} },
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
	runner := New(&fexec, ProtocolIpv4)
	_, err := runner.EnsureRule(Append, TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() call, got %d", fcmd.CombinedOutputCalls)
	}
}

func TestEnsureRuleErrorCreating(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// Status 1 on the first call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{1} },
			// Status 1 on the second call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{1} },
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
	runner := New(&fexec, ProtocolIpv4)
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
			func() ([]byte, error) { return nil, &exec.FakeExitError{1} },
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
	runner := New(&fexec, ProtocolIpv4)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() call, got %d", fcmd.CombinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.CombinedOutputLog[1]...).HasAll("iptables", "-t", "nat", "-C", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
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
	runner := New(&fexec, ProtocolIpv4)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.CombinedOutputLog[2]...).HasAll("iptables", "-t", "nat", "-D", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
}

func TestDeleteRuleErrorChecking(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// iptables version check
			func() ([]byte, error) { return []byte("iptables v1.9.22"), nil },
			// Status 2 on the first call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{2} },
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
	runner := New(&fexec, ProtocolIpv4)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() call, got %d", fcmd.CombinedOutputCalls)
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
			func() ([]byte, error) { return nil, &exec.FakeExitError{1} },
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
	runner := New(&fexec, ProtocolIpv4)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
}

func TestGetIptablesHasCheckCommand(t *testing.T) {
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
		check, err := getIptablesHasCheckCommand(&fexec)
		if (err != nil) != testCase.Err {
			t.Errorf("Expected error: %v, Got error: %v", testCase.Err, err)
		}
		if err == nil && testCase.Expected != check {
			t.Errorf("Expected result: %v, Got result: %v", testCase.Expected, check)
		}
	}
}

func TestExtractIptablesVersion(t *testing.T) {
	testCases := []struct {
		Version string
		V1      int
		V2      int
		V3      int
		Err     bool
	}{
		{"iptables v1.4.7", 1, 4, 7, false},
		{"iptables v1.4.11", 1, 4, 11, false},
		{"iptables v0.2.5", 0, 2, 5, false},
		{"iptables v1.2.3.4.5.6", 1, 2, 3, false},
		{"iptables v1.4", 0, 0, 0, true},
		{"iptables v12345.12345.12345.12344", 12345, 12345, 12345, false},
		{"total junk", 0, 0, 0, true},
	}

	for _, testCase := range testCases {
		v1, v2, v3, err := extractIptablesVersion(testCase.Version)
		if (err != nil) != testCase.Err {
			t.Errorf("Expected error: %v, Got error: %v", testCase.Err, err)
		}
		if err == nil {
			if v1 != testCase.V1 {
				t.Errorf("First version number incorrect for string \"%s\", got %d, expected %d", testCase.Version, v1, testCase.V1)
			}
			if v2 != testCase.V2 {
				t.Errorf("Second version number incorrect for string \"%s\", got %d, expected %d", testCase.Version, v2, testCase.V2)
			}
			if v3 != testCase.V3 {
				t.Errorf("Third version number incorrect for string \"%s\", got %d, expected %d", testCase.Version, v3, testCase.V3)
			}
		}
	}
}

func TestIptablesHasCheckCommand(t *testing.T) {
	testCases := []struct {
		V1     int
		V2     int
		V3     int
		Result bool
	}{
		{0, 55, 55, false},
		{1, 0, 55, false},
		{1, 4, 10, false},
		{1, 4, 11, true},
		{1, 4, 19, true},
		{1, 5, 0, true},
		{2, 0, 0, true},
	}

	for _, testCase := range testCases {
		if result := iptablesHasCheckCommand(testCase.V1, testCase.V2, testCase.V3); result != testCase.Result {
			t.Errorf("For %d.%d.%d expected %v got %v", testCase.V1, testCase.V2, testCase.V3, testCase.Result, result)
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
	if !util.NewStringSet(fcmd.CombinedOutputLog[0]...).HasAll("iptables-save", "-t", "nat") {
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
	if !util.NewStringSet(fcmd.CombinedOutputLog[0]...).HasAll("iptables-save", "-t", "nat") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
	}
}
