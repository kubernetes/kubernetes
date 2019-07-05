/*
Copyright 2017 The Kubernetes Authors.

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
	"context"
	"fmt"
	"io"

	"k8s.io/utils/exec"
)

// FakeExec is a simple scripted Interface type.
type FakeExec struct {
	CommandScript []FakeCommandAction
	CommandCalls  int
	LookPathFunc  func(string) (string, error)
}

var _ exec.Interface = &FakeExec{}

// FakeCommandAction is the function to be executed
type FakeCommandAction func(cmd string, args ...string) exec.Cmd

// Command is to track the commands that are executed
func (fake *FakeExec) Command(cmd string, args ...string) exec.Cmd {
	if fake.CommandCalls > len(fake.CommandScript)-1 {
		panic(fmt.Sprintf("ran out of Command() actions. Could not handle command [%d]: %s args: %v", fake.CommandCalls, cmd, args))
	}
	i := fake.CommandCalls
	fake.CommandCalls++
	return fake.CommandScript[i](cmd, args...)
}

// CommandContext wraps arguments into exec.Cmd
func (fake *FakeExec) CommandContext(ctx context.Context, cmd string, args ...string) exec.Cmd {
	return fake.Command(cmd, args...)
}

// LookPath is for finding the path of a file
func (fake *FakeExec) LookPath(file string) (string, error) {
	return fake.LookPathFunc(file)
}

// FakeCmd is a simple scripted Cmd type.
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
	Env                  []string
	StdoutPipeResponse   FakeStdIOPipeResponse
	StderrPipeResponse   FakeStdIOPipeResponse
	WaitResponse         error
	StartResponse        error
}

var _ exec.Cmd = &FakeCmd{}

// InitFakeCmd is for creating a fake exec.Cmd
func InitFakeCmd(fake *FakeCmd, cmd string, args ...string) exec.Cmd {
	fake.Argv = append([]string{cmd}, args...)
	return fake
}

// FakeStdIOPipeResponse holds responses to use as fakes for the StdoutPipe and
// StderrPipe method calls
type FakeStdIOPipeResponse struct {
	ReadCloser io.ReadCloser
	Error      error
}

// FakeCombinedOutputAction is a function type
type FakeCombinedOutputAction func() ([]byte, error)

// FakeRunAction is a function type
type FakeRunAction func() ([]byte, []byte, error)

// SetDir sets the directory
func (fake *FakeCmd) SetDir(dir string) {
	fake.Dirs = append(fake.Dirs, dir)
}

// SetStdin sets the stdin
func (fake *FakeCmd) SetStdin(in io.Reader) {
	fake.Stdin = in
}

// SetStdout sets the stdout
func (fake *FakeCmd) SetStdout(out io.Writer) {
	fake.Stdout = out
}

// SetStderr sets the stderr
func (fake *FakeCmd) SetStderr(out io.Writer) {
	fake.Stderr = out
}

// SetEnv sets the environment variables
func (fake *FakeCmd) SetEnv(env []string) {
	fake.Env = env
}

// StdoutPipe returns an injected ReadCloser & error (via StdoutPipeResponse)
// to be able to inject an output stream on Stdout
func (fake *FakeCmd) StdoutPipe() (io.ReadCloser, error) {
	return fake.StdoutPipeResponse.ReadCloser, fake.StdoutPipeResponse.Error
}

// StderrPipe returns an injected ReadCloser & error (via StderrPipeResponse)
// to be able to inject an output stream on Stderr
func (fake *FakeCmd) StderrPipe() (io.ReadCloser, error) {
	return fake.StderrPipeResponse.ReadCloser, fake.StderrPipeResponse.Error
}

// Start mimicks starting the process (in the background) and returns the
// injected StartResponse
func (fake *FakeCmd) Start() error {
	return fake.StartResponse
}

// Wait mimicks waiting for the process to exit returns the
// injected WaitResponse
func (fake *FakeCmd) Wait() error {
	return fake.WaitResponse
}

// Run sets runs the command
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

// CombinedOutput returns the output from the command
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

// Output is the response from the command
func (fake *FakeCmd) Output() ([]byte, error) {
	return nil, fmt.Errorf("unimplemented")
}

// Stop is to stop the process
func (fake *FakeCmd) Stop() {
	// no-op
}

// FakeExitError is a simple fake ExitError type.
type FakeExitError struct {
	Status int
}

var _ exec.ExitError = FakeExitError{}

func (fake FakeExitError) String() string {
	return fmt.Sprintf("exit %d", fake.Status)
}

func (fake FakeExitError) Error() string {
	return fake.String()
}

// Exited always returns true
func (fake FakeExitError) Exited() bool {
	return true
}

// ExitStatus returns the fake status
func (fake FakeExitError) ExitStatus() int {
	return fake.Status
}
