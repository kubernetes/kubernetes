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

package exec

import (
	"fmt"
)

// A simple scripted Interface type.
type FakeExec struct {
	CommandScript []FakeCommandAction
	CommandCalls  int
}

type FakeCommandAction func(cmd string, args ...string) Cmd

func (fake *FakeExec) Command(cmd string, args ...string) Cmd {
	if fake.CommandCalls > len(fake.CommandScript)-1 {
		panic("ran out of Command() actions")
	}
	i := fake.CommandCalls
	fake.CommandCalls++
	return fake.CommandScript[i](cmd, args...)
}

// A simple scripted Cmd type.
type FakeCmd struct {
	Argv                 []string
	CombinedOutputScript []FakeCombinedOutputAction
	CombinedOutputCalls  int
	CombinedOutputLog    [][]string
	Dirs                 []string
}

func InitFakeCmd(fake *FakeCmd, cmd string, args ...string) Cmd {
	fake.Argv = append([]string{cmd}, args...)
	return fake
}

type FakeCombinedOutputAction func() ([]byte, error)

func (fake *FakeCmd) SetDir(dir string) {
	fake.Dirs = append(fake.Dirs, dir)
}

func (fake *FakeCmd) CombinedOutput() ([]byte, error) {
	if fake.CombinedOutputCalls > len(fake.CombinedOutputScript)-1 {
		panic("ran out of CombinedOutput() actions")
	}
	if fake.CombinedOutputLog == nil {
		fake.CombinedOutputLog = [][]string{}
	}
	i := fake.CombinedOutputCalls
	fake.CombinedOutputLog = append(fake.CombinedOutputLog, append([]string{}, fake.Argv...))
	fake.CombinedOutputCalls++
	return fake.CombinedOutputScript[i]()
}

// A simple fake ExitError type.
type FakeExitError struct {
	Status int
}

func (fake *FakeExitError) String() string {
	return fmt.Sprintf("exit %d", fake.Status)
}

func (fake *FakeExitError) Error() string {
	return fake.String()
}

func (fake *FakeExitError) Exited() bool {
	return true
}

func (fake *FakeExitError) ExitStatus() int {
	return fake.Status
}
