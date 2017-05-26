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

	"k8s.io/kubernetes/pkg/util/dbus"
	"k8s.io/kubernetes/pkg/util/exec"
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
	cmd := getIPTablesCommand(protocol)
	fexec := exec.NewFakeExec(t, nil)

	// Version checks always use IPv4 binaries
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)

	runner := New(fexec, dbus.NewFake(nil, nil), protocol)
	defer runner.Destroy()

	fexec.AssertExpectedCommands()

	// Success.
	fexec.AddCommand(cmd, "-w2", "-N", "FOOBAR", "-t", "nat").
		SetCombinedOutput("", nil)
	exists, err := runner.EnsureChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if exists {
		t.Errorf("expected exists = false")
	}
	fexec.AssertExpectedCommands()

	// Exists.
	fexec.AddCommand(cmd, "-w2", "-N", "FOOBAR", "-t", "nat").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 1})
	exists, err = runner.EnsureChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if !exists {
		t.Errorf("expected exists = true")
	}
	fexec.AssertExpectedCommands()

	// Failure.
	fexec.AddCommand(cmd, "-w2", "-N", "FOOBAR", "-t", "nat").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 2})
	_, err = runner.EnsureChain(TableNAT, Chain("FOOBAR"))
	if err == nil {
		t.Errorf("expected failure")
	}
	fexec.AssertExpectedCommands()
}

func TestEnsureChainIpv4(t *testing.T) {
	testEnsureChain(t, ProtocolIpv4)
}

func TestEnsureChainIpv6(t *testing.T) {
	testEnsureChain(t, ProtocolIpv6)
}

func TestFlushChain(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)

	runner := New(fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	fexec.AssertExpectedCommands()

	// Success.
	fexec.AddCommand("iptables", "-w2", "-F", "FOOBAR", "-t", "nat").
		SetCombinedOutput("", nil)
	err := runner.FlushChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	fexec.AssertExpectedCommands()

	// Failure.
	fexec.AddCommand("iptables", "-w2", "-F", "FOOBAR", "-t", "nat").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 1})
	err = runner.FlushChain(TableNAT, Chain("FOOBAR"))
	if err == nil {
		t.Errorf("expected failure")
	}
	fexec.AssertExpectedCommands()
}

func TestDeleteChain(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)

	runner := New(fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	fexec.AssertExpectedCommands()

	// Success.
	fexec.AddCommand("iptables", "-w2", "-X", "FOOBAR", "-t", "nat").
		SetCombinedOutput("", nil)
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	fexec.AssertExpectedCommands()

	// Failure.
	fexec.AddCommand("iptables", "-w2", "-X", "FOOBAR", "-t", "nat").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 1})
	err = runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err == nil {
		t.Errorf("expected failure")
	}
	fexec.AssertExpectedCommands()
}

func TestEnsureRuleAlreadyExists(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)

	runner := New(fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	fexec.AssertExpectedCommands()

	fexec.AddCommand("iptables", "-w2", "-C", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput("", nil)
	exists, err := runner.EnsureRule(Append, TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if !exists {
		t.Errorf("expected exists = true")
	}
	fexec.AssertExpectedCommands()
}

func TestEnsureRuleNew(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)

	runner := New(fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	fexec.AssertExpectedCommands()

	fexec.AddCommand("iptables", "-w2", "-C", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 1})
	fexec.AddCommand("iptables", "-w2", "-A", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput("", nil)
	exists, err := runner.EnsureRule(Append, TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if exists {
		t.Errorf("expected exists = false")
	}
	fexec.AssertExpectedCommands()
}

func TestEnsureRuleErrorChecking(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)

	runner := New(fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	fexec.AssertExpectedCommands()

	fexec.AddCommand("iptables", "-w2", "-C", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 2})
	_, err := runner.EnsureRule(Append, TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	fexec.AssertExpectedCommands()
}

func TestEnsureRuleErrorCreating(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)

	runner := New(fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	fexec.AssertExpectedCommands()

	fexec.AddCommand("iptables", "-w2", "-C", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 1})
	fexec.AddCommand("iptables", "-w2", "-A", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 1})
	_, err := runner.EnsureRule(Append, TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	fexec.AssertExpectedCommands()
}

func TestDeleteRuleDoesNotExist(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)

	runner := New(fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	fexec.AssertExpectedCommands()

	fexec.AddCommand("iptables", "-w2", "-C", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 1})
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	fexec.AssertExpectedCommands()
}

func TestDeleteRuleExists(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)

	runner := New(fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	fexec.AssertExpectedCommands()

	fexec.AddCommand("iptables", "-w2", "-C", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput("", nil)
	fexec.AddCommand("iptables", "-w2", "-D", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput("", nil)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	fexec.AssertExpectedCommands()
}

func TestDeleteRuleErrorChecking(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)

	runner := New(fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	fexec.AssertExpectedCommands()

	fexec.AddCommand("iptables", "-w2", "-C", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 2})
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	fexec.AssertExpectedCommands()
}

func TestDeleteRuleErrorDeleting(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)

	runner := New(fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	fexec.AssertExpectedCommands()

	fexec.AddCommand("iptables", "-w2", "-C", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput("", nil)
	fexec.AddCommand("iptables", "-w2", "-D", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 1})
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	fexec.AssertExpectedCommands()
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
		fexec := exec.NewFakeExec(t, nil)
		fexec.AddCommand("iptables", "--version").
			SetCombinedOutput(testCase.Version, nil)

		version, err := getIPTablesVersionString(fexec)
		if (err != nil) != testCase.Err {
			t.Errorf("Expected error: %v, Got error: %v", testCase.Err, err)
		}
		if err == nil {
			check := getIPTablesHasCheckCommand(version)
			if testCase.Expected != check {
				t.Errorf("Expected result: %v, Got result: %v", testCase.Expected, check)
			}
		}
		fexec.AssertExpectedCommands()
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

	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables-save", "-t", "nat").
		SetCombinedOutput(iptables_save_output, nil)
	defer fexec.AssertExpectedCommands()

	runner := &runner{exec: fexec}
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

	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables-save", "-t", "nat").
		SetCombinedOutput(iptables_save_output, nil)
	defer fexec.AssertExpectedCommands()

	runner := &runner{exec: fexec}
	exists, err := runner.checkRuleWithoutCheck(TableNAT, ChainPrerouting, "-m", "addrtype", "-j", "DOCKER")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if exists {
		t.Errorf("expected exists = false")
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
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.4.19", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)
	fexec.AddCommand("iptables", "-X", "FOOBAR", "-t", "nat").
		SetCombinedOutput("", nil)
	defer fexec.AssertExpectedCommands()

	runner := New(fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
}

func TestWaitFlagOld(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.4.20", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)
	fexec.AddCommand("iptables", "-w", "-X", "FOOBAR", "-t", "nat").
		SetCombinedOutput("", nil)
	defer fexec.AssertExpectedCommands()

	runner := New(fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
}

func TestWaitFlagNew(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.4.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)
	fexec.AddCommand("iptables", "-w2", "-X", "FOOBAR", "-t", "nat").
		SetCombinedOutput("", nil)
	defer fexec.AssertExpectedCommands()

	runner := New(fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	err := runner.DeleteChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
}

func TestReload(t *testing.T) {
	dbusConn := dbus.NewFakeConnection()
	dbusConn.SetBusObject(func(method string, args ...interface{}) ([]interface{}, error) { return nil, nil })
	dbusConn.AddObject(firewalldName, firewalldPath, func(method string, args ...interface{}) ([]interface{}, error) { return nil, nil })
	fdbus := dbus.NewFake(dbusConn, nil)

	reloaded := make(chan bool, 2)

	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.4.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)

	runner := New(fexec, fdbus, ProtocolIpv4)
	defer runner.Destroy()
	fexec.AssertExpectedCommands()

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

	// first reload
	fexec.AddCommand("iptables", "-w2", "-N", "FOOBAR", "-t", "nat").
		SetCombinedOutput("", nil)
	fexec.AddCommand("iptables", "-w2", "-C", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 1})
	fexec.AddCommand("iptables", "-w2", "-A", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput("", nil)

	dbusConn.EmitSignal("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "NameOwnerChanged", firewalldName, "", ":1.1")
	<-reloaded
	<-reloaded
	fexec.AssertExpectedCommands()

	go func() { time.Sleep(time.Second / 100); reloaded <- true }()
	dbusConn.EmitSignal(firewalldName, firewalldPath, firewalldInterface, "DefaultZoneChanged", "public")
	dbusConn.EmitSignal("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "NameOwnerChanged", "io.k8s.Something", "", ":1.1")
	<-reloaded
	// If those signals caused a reload, FakeExec would panic because it doesn't have any commands to run

	// second reload
	fexec.AddCommand("iptables", "-w2", "-N", "FOOBAR", "-t", "nat").
		SetCombinedOutput("", nil)
	fexec.AddCommand("iptables", "-w2", "-C", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 1})
	fexec.AddCommand("iptables", "-w2", "-A", "OUTPUT", "-t", "nat", "abc", "123").
		SetCombinedOutput("", nil)

	dbusConn.EmitSignal(firewalldName, firewalldPath, firewalldInterface, "Reloaded")
	<-reloaded
	<-reloaded
	fexec.AssertExpectedCommands()
}

func TestSaveInto(t *testing.T) {
	output := `# Generated by iptables-save v1.6.0 on Thu Jan 19 11:38:09 2017
*filter
:INPUT ACCEPT [15079:38410730]
:FORWARD ACCEPT [0:0]
:OUTPUT ACCEPT [11045:521562]
COMMIT
# Completed on Thu Jan 19 11:38:09 2017`

	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)

	runner := New(fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	fexec.AssertExpectedCommands()

	buffer := bytes.NewBuffer(nil)

	// Success.
	fexec.AddCommand("iptables-save", "-t", "nat").
		SetRunOutput(output, nil, nil)
	err := runner.SaveInto(TableNAT, buffer)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	fexec.AssertExpectedCommands()

	if string(buffer.Bytes()[:len(output)]) != output {
		t.Errorf("expected output to be equal to mocked one, got %v", buffer.Bytes())
	}

	// Failure.
	buffer.Reset()
	fexec.AddCommand("iptables-save", "-t", "nat").
		SetRunOutput(nil, nil, &exec.FakeExitError{Status: 1})
	err = runner.SaveInto(TableNAT, buffer)
	if err == nil {
		t.Errorf("expected failure")
	}
	fexec.AssertExpectedCommands()
}

func TestRestore(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)

	runner := New(fexec, dbus.NewFake(nil, nil), ProtocolIpv4)
	defer runner.Destroy()
	fexec.AssertExpectedCommands()

	// both flags true
	fexec.AddCommand("iptables-restore", "--wait=2", "-T", "nat", "--counters").
		SetCombinedOutput("", nil)
	err := runner.Restore(TableNAT, []byte{}, FlushTables, RestoreCounters)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	fexec.AssertExpectedCommands()

	// FlushTables, NoRestoreCounters
	fexec.AddCommand("iptables-restore", "--wait=2", "-T", "nat").
		SetCombinedOutput("", nil)
	err = runner.Restore(TableNAT, []byte{}, FlushTables, NoRestoreCounters)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	fexec.AssertExpectedCommands()

	// NoFlushTables, RestoreCounters
	fexec.AddCommand("iptables-restore", "--wait=2", "-T", "nat", "--noflush", "--counters").
		SetCombinedOutput("", nil)
	err = runner.Restore(TableNAT, []byte{}, NoFlushTables, RestoreCounters)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	fexec.AssertExpectedCommands()

	// NoFlushTables, NoRestoreCounters
	fexec.AddCommand("iptables-restore", "--wait=2", "-T", "nat", "--noflush").
		SetCombinedOutput("", nil)
	err = runner.Restore(TableNAT, []byte{}, NoFlushTables, NoRestoreCounters)
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	fexec.AssertExpectedCommands()

	// Failure.
	fexec.AddCommand("iptables-restore", "--wait=2", "-T", "nat", "--counters").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 1})
	err = runner.Restore(TableNAT, []byte{}, FlushTables, RestoreCounters)
	if err == nil {
		t.Errorf("expected failure")
	}
	fexec.AssertExpectedCommands()
}

// TestRestoreAll tests only the simplest use case, as flag handling code is already tested in TestRestore
func TestRestoreAll(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("iptables-restore v1.9.22", nil)

	runner := newInternal(fexec, dbus.NewFake(nil, nil), ProtocolIpv4, TestLockfilePath)
	defer os.Remove(TestLockfilePath)
	defer runner.Destroy()
	fexec.AssertExpectedCommands()

	fexec.AddCommand("iptables-restore", "--wait=2", "--noflush", "--counters").
		SetCombinedOutput("", nil)
	err := runner.RestoreAll([]byte{}, NoFlushTables, RestoreCounters)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	fexec.AssertExpectedCommands()

	// Failure.
	fexec.AddCommand("iptables-restore", "--wait=2", "--counters").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 1})
	err = runner.RestoreAll([]byte{}, FlushTables, RestoreCounters)
	if err == nil {
		t.Errorf("expected failure")
	}
	fexec.AssertExpectedCommands()
}

// TestRestoreAllWaitOldIptablesRestore tests that the "wait" flag is not passed
// to a in-compatible iptables-restore
func TestRestoreAllWaitOldIptablesRestore(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("unrecognized option: --version", nil)

	runner := newInternal(fexec, dbus.NewFake(nil, nil), ProtocolIpv4, TestLockfilePath)
	defer os.Remove(TestLockfilePath)
	defer runner.Destroy()
	fexec.AssertExpectedCommands()

	fexec.AddCommand("iptables-restore", "--noflush", "--counters").
		SetCombinedOutput("", nil)
	err := runner.RestoreAll([]byte{}, NoFlushTables, RestoreCounters)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	fexec.AssertExpectedCommands()

	// Failure.
	fexec.AddCommand("iptables-restore", "-T", "nat", "--counters").
		SetCombinedOutput(nil, &exec.FakeExitError{Status: 1})
	err = runner.Restore(TableNAT, []byte{}, FlushTables, RestoreCounters)
	if err == nil {
		t.Errorf("expected failure")
	}
	fexec.AssertExpectedCommands()
}

// TestRestoreAllGrabNewLock tests that the iptables code will grab the
// iptables /run lock when using an iptables-restore version that does not
// support the --wait argument
func TestRestoreAllGrabNewLock(t *testing.T) {
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("unrecognized option: --version", nil)
	defer fexec.AssertExpectedCommands()

	runner := newInternal(fexec, dbus.NewFake(nil, nil), ProtocolIpv4, TestLockfilePath)
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
	fexec := exec.NewFakeExec(t, nil)
	fexec.AddCommand("iptables", "--version").
		SetCombinedOutput("iptables v1.9.22", nil)
	fexec.AddCommand("iptables-restore", "--version").
		SetCombinedOutput("unrecognized option: --version", nil)
	defer fexec.AssertExpectedCommands()

	runner := newInternal(fexec, dbus.NewFake(nil, nil), ProtocolIpv4, TestLockfilePath)
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
