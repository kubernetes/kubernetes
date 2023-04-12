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
	"sync"

	"k8s.io/utils/exec"
)

// FakeExec is a simple scripted Interface type.
type FakeExec struct {
	CommandScript []FakeCommandAction
	CommandCalls  int
	LookPathFunc  func(string) (string, error)
	// ExactOrder enforces that commands are called in the order they are scripted,
	// and with the exact same arguments
	ExactOrder bool
	// DisableScripts removes the requirement that CommandScripts be populated
	// before calling Command(). This makes Command() and subsequent calls to
	// Run() or CombinedOutput() always return success and empty output.
	DisableScripts bool

	mu sync.Mutex
}

var _ exec.Interface = &FakeExec{}

// FakeCommandAction is the function to be executed
type FakeCommandAction func(cmd string, args ...string) exec.Cmd

// Command returns the next unexecuted command in CommandScripts.
// This function is safe for concurrent access as long as the underlying
// FakeExec struct is not modified during execution.
func (fake *FakeExec) Command(cmd string, args ...string) exec.Cmd {
	if fake.DisableScripts {
		fakeCmd := &FakeCmd{DisableScripts: true}
		return InitFakeCmd(fakeCmd, cmd, args...)
	}
	fakeCmd := fake.nextCommand(cmd, args)
	if fake.ExactOrder {
		argv := append([]string{cmd}, args...)
		fc := fakeCmd.(*FakeCmd)
		if cmd != fc.Argv[0] {
			panic(fmt.Sprintf("received command: %s, expected: %s", cmd, fc.Argv[0]))
		}
		if len(argv) != len(fc.Argv) {
			panic(fmt.Sprintf("command (%s) received with extra/missing arguments. Expected %v, Received %v", cmd, fc.Argv, argv))
		}
		for i, a := range argv[1:] {
			if a != fc.Argv[i+1] {
				panic(fmt.Sprintf("command (%s) called with unexpected argument. Expected %s, Received %s", cmd, fc.Argv[i+1], a))
			}
		}
	}
	return fakeCmd
}

func (fake *FakeExec) nextCommand(cmd string, args []string) exec.Cmd {
	fake.mu.Lock()
	defer fake.mu.Unlock()

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
	CombinedOutputScript []FakeAction
	CombinedOutputCalls  int
	CombinedOutputLog    [][]string
	OutputScript         []FakeAction
	OutputCalls          int
	OutputLog            [][]string
	RunScript            []FakeAction
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
	DisableScripts       bool
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

// FakeAction is a function type
type FakeAction func() ([]byte, []byte, error)

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

// Run runs the command
func (fake *FakeCmd) Run() error {
	if fake.DisableScripts {
		return nil
	}
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
	if fake.DisableScripts {
		return []byte{}, nil
	}
	if fake.CombinedOutputCalls > len(fake.CombinedOutputScript)-1 {
		panic("ran out of CombinedOutput() actions")
	}
	if fake.CombinedOutputLog == nil {
		fake.CombinedOutputLog = [][]string{}
	}
	i := fake.CombinedOutputCalls
	fake.CombinedOutputLog = append(fake.CombinedOutputLog, append([]string{}, fake.Argv...))
	fake.CombinedOutputCalls++
	stdout, _, err := fake.CombinedOutputScript[i]()
	return stdout, err
}

// Output is the response from the command
func (fake *FakeCmd) Output() ([]byte, error) {
	if fake.DisableScripts {
		return []byte{}, nil
	}
	if fake.OutputCalls > len(fake.OutputScript)-1 {
		panic("ran out of Output() actions")
	}
	if fake.OutputLog == nil {
		fake.OutputLog = [][]string{}
	}
	i := fake.OutputCalls
	fake.OutputLog = append(fake.OutputLog, append([]string{}, fake.Argv...))
	fake.OutputCalls++
	stdout, _, err := fake.OutputScript[i]()
	return stdout, err
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
