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

package kubelet

import (
	"testing"

	"k8s.io/kubernetes/pkg/util/dbus"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/iptables"
	"k8s.io/kubernetes/pkg/util/sets"
)

func TestEnsureIPTablesMasqRuleNew(t *testing.T) {
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
	runner := iptables.New(&fexec, dbus.NewFake(nil, nil), iptables.ProtocolIpv4)
	defer runner.Destroy()
	err := ensureIPTablesMasqRule(runner, "127.0.0.0/8")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[2]...).HasAll("iptables", "-t", "nat", "-A", "POSTROUTING",
		"-m", "comment", "--comment", "kubelet: SNAT outbound cluster traffic",
		"!", "-d", "127.0.0.0/8", "-j", "MASQUERADE") {
		t.Errorf("wrong CombinedOutput() log, got %#v", fcmd.CombinedOutputLog[2])
	}
}

func TestEnsureIPTablesMasqRuleAlreadyExists(t *testing.T) {
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
	runner := iptables.New(&fexec, dbus.NewFake(nil, nil), iptables.ProtocolIpv4)
	defer runner.Destroy()
	err := ensureIPTablesMasqRule(runner, "127.0.0.0/8")
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !sets.NewString(fcmd.CombinedOutputLog[1]...).HasAll("iptables", "-t", "nat", "-C", "POSTROUTING",
		"-m", "comment", "--comment", "kubelet: SNAT outbound cluster traffic",
		"!", "-d", "127.0.0.0/8", "-j", "MASQUERADE") {
		t.Errorf("wrong CombinedOutput() log, got %#v", fcmd.CombinedOutputLog[1])
	}
}

func TestEnsureIPTablesMasqRuleErrorChecking(t *testing.T) {
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
	runner := iptables.New(&fexec, dbus.NewFake(nil, nil), iptables.ProtocolIpv4)
	defer runner.Destroy()
	err := ensureIPTablesMasqRule(runner, "127.0.0.0/8")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
}

func TestEnsureIPTablesMasqRuleErrorCreating(t *testing.T) {
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
	runner := iptables.New(&fexec, dbus.NewFake(nil, nil), iptables.ProtocolIpv4)
	defer runner.Destroy()
	err := ensureIPTablesMasqRule(runner, "127.0.0.0/8")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 3 {
		t.Errorf("expected 3 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
}
