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
	"fmt"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	utilexec "github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
)

// A simple scripted utilexec.Interface type.
type fakeExec struct {
	commandScript []fakeCommandAction
	commandCalls  int
}

type fakeCommandAction func(cmd string, args ...string) utilexec.Cmd

func (fake *fakeExec) Command(cmd string, args ...string) utilexec.Cmd {
	if fake.commandCalls > len(fake.commandScript)-1 {
		panic("ran out of Command() actions")
	}
	i := fake.commandCalls
	fake.commandCalls++
	return fake.commandScript[i](cmd, args...)
}

// A simple scripted utilexec.Cmd type.
type fakeCmd struct {
	argv                 []string
	combinedOutputScript []fakeCombinedOutputAction
	combinedOutputCalls  int
	combinedOutputLog    [][]string
}

func initFakeCmd(fake *fakeCmd, cmd string, args ...string) utilexec.Cmd {
	fake.argv = append([]string{cmd}, args...)
	return fake
}

type fakeCombinedOutputAction func() ([]byte, error)

func (fake *fakeCmd) CombinedOutput() ([]byte, error) {
	if fake.combinedOutputCalls > len(fake.combinedOutputScript)-1 {
		panic("ran out of CombinedOutput() actions")
	}
	if fake.combinedOutputLog == nil {
		fake.combinedOutputLog = [][]string{}
	}
	i := fake.combinedOutputCalls
	fake.combinedOutputLog = append(fake.combinedOutputLog, append([]string{}, fake.argv...))
	fake.combinedOutputCalls++
	return fake.combinedOutputScript[i]()
}

// A simple fake utilexec.ExitError type.
type fakeExitError struct {
	status int
}

func (fake *fakeExitError) String() string {
	return fmt.Sprintf("exit %d", fake.status)
}

func (fake *fakeExitError) Error() string {
	return fake.String()
}

func (fake *fakeExitError) Exited() bool {
	return true
}

func (fake *fakeExitError) ExitStatus() int {
	return fake.status
}

func TestEnsureChain(t *testing.T) {
	fcmd := fakeCmd{
		combinedOutputScript: []fakeCombinedOutputAction{
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
			// Exists.
			func() ([]byte, error) { return nil, &fakeExitError{1} },
			// Failure.
			func() ([]byte, error) { return nil, &fakeExitError{2} },
		},
	}
	fexec := fakeExec{
		commandScript: []fakeCommandAction{
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
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
	if fcmd.combinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.combinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.combinedOutputLog[0]...).HasAll("iptables", "-t", "nat", "-N", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.combinedOutputLog[0])
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
	fcmd := fakeCmd{
		combinedOutputScript: []fakeCombinedOutputAction{
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
			// Failure.
			func() ([]byte, error) { return nil, &fakeExitError{1} },
		},
	}
	fexec := fakeExec{
		commandScript: []fakeCommandAction{
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	// Success.
	err := runner.FlushChain(TableNAT, Chain("FOOBAR"))
	if err != nil {
		t.Errorf("expected success, got %+v", err)
	}
	if fcmd.combinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.combinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.combinedOutputLog[0]...).HasAll("iptables", "-t", "nat", "-F", "FOOBAR") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.combinedOutputLog[0])
	}
	// Failure.
	err = runner.FlushChain(TableNAT, Chain("FOOBAR"))
	if err == nil {
		t.Errorf("expected failure")
	}
}

func TestEnsureRuleAlreadyExists(t *testing.T) {
	fcmd := fakeCmd{
		combinedOutputScript: []fakeCombinedOutputAction{
			// Success.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := fakeExec{
		commandScript: []fakeCommandAction{
			// The first Command() call is checking the rule.  Success of that exec means "done".
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
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
	if fcmd.combinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.combinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.combinedOutputLog[0]...).HasAll("iptables", "-t", "nat", "-C", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.combinedOutputLog[0])
	}
}

func TestEnsureRuleNew(t *testing.T) {
	fcmd := fakeCmd{
		combinedOutputScript: []fakeCombinedOutputAction{
			// Status 1 on the first call.
			func() ([]byte, error) { return nil, &fakeExitError{1} },
			// Success on the second call.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := fakeExec{
		commandScript: []fakeCommandAction{
			// The first Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
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
	if fcmd.combinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.combinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.combinedOutputLog[1]...).HasAll("iptables", "-t", "nat", "-A", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.combinedOutputLog[1])
	}
}

func TestEnsureRuleErrorChecking(t *testing.T) {
	fcmd := fakeCmd{
		combinedOutputScript: []fakeCombinedOutputAction{
			// Status 2 on the first call.
			func() ([]byte, error) { return nil, &fakeExitError{2} },
		},
	}
	fexec := fakeExec{
		commandScript: []fakeCommandAction{
			// The first Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	_, err := runner.EnsureRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.combinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.combinedOutputCalls)
	}
}

func TestEnsureRuleErrorCreating(t *testing.T) {
	fcmd := fakeCmd{
		combinedOutputScript: []fakeCombinedOutputAction{
			// Status 1 on the first call.
			func() ([]byte, error) { return nil, &fakeExitError{1} },
			// Status 1 on the second call.
			func() ([]byte, error) { return nil, &fakeExitError{1} },
		},
	}
	fexec := fakeExec{
		commandScript: []fakeCommandAction{
			// The first Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	_, err := runner.EnsureRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.combinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.combinedOutputCalls)
	}
}

func TestDeleteRuleAlreadyExists(t *testing.T) {
	fcmd := fakeCmd{
		combinedOutputScript: []fakeCombinedOutputAction{
			// Status 1 on the first call.
			func() ([]byte, error) { return nil, &fakeExitError{1} },
		},
	}
	fexec := fakeExec{
		commandScript: []fakeCommandAction{
			// The first Command() call is checking the rule.  Failure of that exec means "does not exist".
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %+v", err)
	}
	if fcmd.combinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.combinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.combinedOutputLog[0]...).HasAll("iptables", "-t", "nat", "-C", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.combinedOutputLog[0])
	}
}

func TestDeleteRuleNew(t *testing.T) {
	fcmd := fakeCmd{
		combinedOutputScript: []fakeCombinedOutputAction{
			// Success on the first call.
			func() ([]byte, error) { return []byte{}, nil },
			// Success on the second call.
			func() ([]byte, error) { return []byte{}, nil },
		},
	}
	fexec := fakeExec{
		commandScript: []fakeCommandAction{
			// The first Command() call is checking the rule.  Success of that means delete it.
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err != nil {
		t.Errorf("expected success, got %+v", err)
	}
	if fcmd.combinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.combinedOutputCalls)
	}
	if !util.NewStringSet(fcmd.combinedOutputLog[1]...).HasAll("iptables", "-t", "nat", "-D", "OUTPUT", "abc", "123") {
		t.Errorf("wrong CombinedOutput() log, got %s", fcmd.combinedOutputLog[1])
	}
}

func TestDeleteRuleErrorChecking(t *testing.T) {
	fcmd := fakeCmd{
		combinedOutputScript: []fakeCombinedOutputAction{
			// Status 2 on the first call.
			func() ([]byte, error) { return nil, &fakeExitError{2} },
		},
	}
	fexec := fakeExec{
		commandScript: []fakeCommandAction{
			// The first Command() call is checking the rule.  Failure of that means create it.
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.combinedOutputCalls != 1 {
		t.Errorf("expected 1 CombinedOutput() call, got %d", fcmd.combinedOutputCalls)
	}
}

func TestDeleteRuleErrorCreating(t *testing.T) {
	fcmd := fakeCmd{
		combinedOutputScript: []fakeCombinedOutputAction{
			// Success on the first call.
			func() ([]byte, error) { return []byte{}, nil },
			// Status 1 on the second call.
			func() ([]byte, error) { return nil, &fakeExitError{1} },
		},
	}
	fexec := fakeExec{
		commandScript: []fakeCommandAction{
			// The first Command() call is checking the rule.  Success of that means delete it.
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) utilexec.Cmd { return initFakeCmd(&fcmd, cmd, args...) },
		},
	}
	runner := New(&fexec)
	err := runner.DeleteRule(TableNAT, ChainOutput, "abc", "123")
	if err == nil {
		t.Errorf("expected failure")
	}
	if fcmd.combinedOutputCalls != 2 {
		t.Errorf("expected 2 CombinedOutput() calls, got %d", fcmd.combinedOutputCalls)
	}
}
