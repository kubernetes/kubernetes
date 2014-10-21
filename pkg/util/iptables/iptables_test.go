/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
)

func TestEnsureChain(t *testing.T) {
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
	runner := New(&fexec)
	// Success.
	exists, err := runner.EnsureChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %+v", err)
	}
	if exists {
		t.Errorf("expected exists = false")
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.CombinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.CombinedOutputLog[0]...).HasAll("iptables", "-t", "nat", "-N", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
	}
	// Exists.
	exists, err = runner.EnsureChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %+v", err)
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
	runner := New(&fexec)
	// Success.
	err := runner.FlushChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %+v", err)
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

func TestEnsureRuleAlreadyExists(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// The first Command() call is checking the rule.  Success of that exec means "done".
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	exists, err := runner.EnsureRule(TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %+v", err)
	}
	if !exists {
		t.Errorf("expected exists = true")
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.CombinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.CombinedOutputLog[0]...).HasAll("iptables", "-t", "nat", "-C", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
	}
}

func TestEnsureRuleNew(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// Status 1 on the first call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{1} },
			// Success on the second call.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// The first Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	exists, err := runner.EnsureRule(TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %+v", err)
	}
	if exists {
		t.Errorf("expected exists = false")
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.CombinedOutputLog[1]...).HasAll("iptables", "-t", "nat", "-A", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
}

func TestEnsureRuleErrorChecking(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// Status 2 on the first call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{2} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// The first Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	_, err := runner.EnsureRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.CombinedOutputCalls)
	}
}

func TestEnsureRuleErrorCreating(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// Status 1 on the first call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{1} },
			// Status 1 on the second call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{1} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// The first Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	_, err := runner.EnsureRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
}

func TestDeleteRuleAlreadyExists(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// Status 1 on the first call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{1} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// The first Command() call is checking the rule.  Failure of that exec means "does not exist".
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %+v", err)
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.CombinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.CombinedOutputLog[0]...).HasAll("iptables", "-t", "nat", "-C", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[0])
	}
}

func TestDeleteRuleNew(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// Success on the first call.
			func() ([]byte, error) { return []byte{}, nil },
			// Success on the second call.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// The first Command() call is checking the rule.  Success of that means delete it.
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %+v", err)
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.CombinedOutputLog[1]...).HasAll("iptables", "-t", "nat", "-D", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.CombinedOutputLog[1])
	}
}

func TestDeleteRuleErrorChecking(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// Status 2 on the first call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{2} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// The first Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.CombinedOutputCalls)
	}
}

func TestDeleteRuleErrorCreating(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// Success on the first call.
			func() ([]byte, error) { return []byte{}, nil },
			// Status 1 on the second call.
			func() ([]byte, error) { return nil, &exec.FakeExitError{1} },
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			// The first Command() call is checking the rule.  Success of that means delete it.
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.CombinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.CombinedOutputCalls)
	}
}
