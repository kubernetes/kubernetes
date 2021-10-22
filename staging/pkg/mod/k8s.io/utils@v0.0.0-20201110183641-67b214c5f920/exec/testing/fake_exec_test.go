/*
Copyright 2019 The Kubernetes Authors.

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

package testingexec

import (
	"testing"

	"k8s.io/utils/exec"
)

// Test that command order is enforced
func TestCommandOrder(t *testing.T) {
	fe := getFakeExecWithScripts(true, false)

	// If we call "cat" first, it should panic
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()
	fe.Command("cat")
}

// Test that a command with different number of args panics
func TestDiffNumArgs(t *testing.T) {
	fe := getFakeExecWithScripts(true, false)

	// If we call "ps -e -f -A" instead of "ps -ef" it should panic
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()
	fe.Command("ps", "-e", "-f", "-A")
}

// Test that a command with different args panics
func TestDiffArgs(t *testing.T) {
	fe := getFakeExecWithScripts(true, false)

	// If we call "ps -fe" instead of "ps -ef" it should panic
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()
	fe.Command("ps", "-fe")
}

// Test that extra commands panics
func TestExtraCommands(t *testing.T) {
	fe := getFakeExecWithScripts(false, false)

	// If we call mnore calls than scripted, should panic
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()
	fe.Command("ps")
	fe.Command("cat")
	fe.Command("unscripted")
}

// Test that calling a command without a script panics if not disabled
func TestNoScriptPanic(t *testing.T) {
	fe := &FakeExec{}

	// If we call mnore calls than scripted, should panic
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()
	fe.Command("ps")
}

// Test that calling a command without a script does not panic if scripts
// are disabled
func TestNoScriptNoPanic(t *testing.T) {
	fe := &FakeExec{DisableScripts: true}

	// If we call mnore calls than scripted, should panic
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("The code panic'd")
		}
	}()
	fe.Command("ps")
}

func getFakeExecWithScripts(exactOrder bool, disableScripts bool) *FakeExec {
	scripts := []struct {
		cmd  string
		args []string
	}{
		{
			cmd:  "ps",
			args: []string{"-ef"},
		},
		{
			cmd:  "cat",
			args: []string{"/var/log"},
		},
	}

	fakeexec := &FakeExec{ExactOrder: exactOrder, DisableScripts: disableScripts}
	for _, s := range scripts {
		fakeCmd := &FakeCmd{}
		cmdAction := makeFakeCmd(fakeCmd, s.cmd, s.args...)
		fakeexec.CommandScript = append(fakeexec.CommandScript, cmdAction)
	}
	return fakeexec
}

func makeFakeCmd(fakeCmd *FakeCmd, cmd string, args ...string) FakeCommandAction {
	c := cmd
	a := args
	return func(cmd string, args ...string) exec.Cmd {
		command := InitFakeCmd(fakeCmd, c, a...)
		return command
	}
}
