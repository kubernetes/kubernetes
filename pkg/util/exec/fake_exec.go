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

package exec

import (
	"fmt"
	"io"
	osexec "os/exec"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// A simple scripted Interface type.
type FakeExec struct {
	CommandScript []FakeCommandAction
	CommandCalls  int
	LookPathFunc  func(string) (string, error)

	T *testing.T
}

// NewFakeExec creates a FakeExec. You can pass nil for t if you are not running from a
// test program. (This will cause certain failures to result in a panic rather than a
// t.Fatal.) cmdsInPath is a list of command names for which LookPath will return a match
// ("/fake-bin/commandname"); if your code does not use LookPath you can pass nil. (If you
// need more complex behavior, you can override LookPathFunc on the returned FakeExec.)
func NewFakeExec(t *testing.T, cmdsInPath []string) *FakeExec {
	return &FakeExec{
		T: t,
		LookPathFunc: func(cmd string) (string, error) {
			for _, pathCmd := range cmdsInPath {
				if pathCmd == cmd {
					return "/fake-bin/" + cmd, nil
				}
			}
			return "", &osexec.Error{cmd, osexec.ErrNotFound}
		},
	}
}

type FakeCommandAction func(cmd string, args ...string) Cmd

func (fake *FakeExec) Command(cmd string, args ...string) Cmd {
	if fake.CommandCalls > len(fake.CommandScript)-1 {
		fake.fail("ran out of Command() actions at %s executing %v", getCaller(1), append([]string{cmd}, args...))
	}
	i := fake.CommandCalls
	fake.CommandCalls++
	return fake.CommandScript[i](cmd, args...)
}

func (fake *FakeExec) LookPath(file string) (string, error) {
	return fake.LookPathFunc(file)
}

func getCaller(n int) string {
	_, file, line, ok := runtime.Caller(n + 1)
	if !ok {
		return "???"
	}
	if i := strings.LastIndexAny(file, "/\\"); i >= 0 {
		file = file[i+1:]
	}
	return fmt.Sprintf("%s:%d", file, line)
}

func (fake *FakeExec) fail(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	if fake.T == nil {
		panic(msg)
	}

	// Don't re-fail from a "defer fexec.AssertExpectedCommands()" if we already failed once
	if fake.T.Failed() {
		return
	}
	fake.T.Fatal(msg)
}

// AddCommand adds a Cmd to be returned by a future fake.Command(...) call. Call
// SetCombinedOutput or SetRunOutput on the result to specify its result.
func (fake *FakeExec) AddCommand(cmd string, args ...string) *FakeCmd {
	expectCaller := getCaller(1)
	expectedArgv := append([]string{cmd}, args...)
	fcmd := &FakeCmd{}
	fake.CommandScript = append(fake.CommandScript,
		func(cmd string, args ...string) Cmd {
			InitFakeCmd(fcmd, cmd, args...)
			if !reflect.DeepEqual(expectedArgv, fcmd.Argv) {
				fake.fail("Wrong command: expected\n%v (at %s)\ngot\n%v (at %s)", expectedArgv, expectCaller, fcmd.Argv, getCaller(2))
			}
			return fcmd
		},
	)
	return fcmd
}

func outputAsBytes(output interface{}) []byte {
	if outputBytes, ok := output.([]byte); ok {
		return outputBytes
	} else if outputStr, ok := output.(string); ok {
		return []byte(outputStr)
	} else if output == nil {
		return nil
	}
	panic("output must be []byte or string")
}

// SetRunOutput provides the result of calling Run() on a FakeCmd.
// stdout and stderr can be either []byte or string (which will be converted to []byte).
func (fcmd *FakeCmd) SetRunOutput(stdout, stderr interface{}, err error) {
	fcmd.RunScript = []FakeRunAction{
		func() ([]byte, []byte, error) { return outputAsBytes(stdout), outputAsBytes(stderr), err },
	}
}

// SetCombinedOutput provides the result of calling CombinedOutput() on a FakeCmd.
// output can be either a []byte or a string (which will be converted to a []byte).
func (fcmd *FakeCmd) SetCombinedOutput(output interface{}, err error) {
	fcmd.CombinedOutputScript = []FakeCombinedOutputAction{
		func() ([]byte, error) { return outputAsBytes(output), err },
	}
}

// AssertExpectedCommands ensures that all of the commands added to fake via AddCommand were actually executed
func (fake *FakeExec) AssertExpectedCommands() {
	if fake.CommandCalls != len(fake.CommandScript) {
		fake.fail("Only used %d of %d expected commands (at %s)", fake.CommandCalls, len(fake.CommandScript), getCaller(1))
	}
}

// A simple scripted Cmd type.
type FakeCmd struct {
	Argv                 []string
	CombinedOutputScript []FakeCombinedOutputAction
	CombinedOutputCalls  int
	CombinedOutputLog    [][]string
	RunScript            []FakeRunAction
	RunCalls             int
	RunLog               [][]string
	Dirs                 []string
	Stdin                io.Reader
	Stdout               io.Writer
	Stderr               io.Writer
}

func InitFakeCmd(fake *FakeCmd, cmd string, args ...string) Cmd {
	fake.Argv = append([]string{cmd}, args...)
	return fake
}

type FakeCombinedOutputAction func() ([]byte, error)
type FakeRunAction func() ([]byte, []byte, error)

func (fake *FakeCmd) SetDir(dir string) {
	fake.Dirs = append(fake.Dirs, dir)
}

func (fake *FakeCmd) SetStdin(in io.Reader) {
	fake.Stdin = in
}

func (fake *FakeCmd) SetStdout(out io.Writer) {
	fake.Stdout = out
}

func (fake *FakeCmd) SetStderr(out io.Writer) {
	fake.Stderr = out
}

func (fake *FakeCmd) Run() error {
	if fake.RunCalls > len(fake.RunScript)-1 {
		panic("ran out of Run() actions")
	}
	if fake.RunLog == nil {
		fake.RunLog = [][]string{}
	}
	i := fake.RunCalls
	fake.RunLog = append(fake.RunLog, append([]string{}, fake.Argv...))
	fake.RunCalls++
	stdout, stderr, err := fake.RunScript[i]()
	if stdout != nil {
		fake.Stdout.Write(stdout)
	}
	if stderr != nil {
		fake.Stderr.Write(stderr)
	}
	return err
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

func (fake *FakeCmd) Output() ([]byte, error) {
	return nil, fmt.Errorf("unimplemented")
}

func (fake *FakeCmd) Stop() {
	// no-op
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
