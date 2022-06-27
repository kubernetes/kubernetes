/*
Copyright 2015 The Kubernetes Authors.

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
	"errors"
	"github.com/stretchr/testify/assert"
	"io"
	"k8s.io/client-go/util/exec"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/probe"
)

type FakeCmd struct {
	out    []byte
	stdout []byte
	err    error
	writer io.Writer
}

func (f *FakeCmd) Run() error {
	return nil
}

func (f *FakeCmd) CombinedOutput() ([]byte, error) {
	return f.out, f.err
}

func (f *FakeCmd) Output() ([]byte, error) {
	return f.stdout, f.err
}

func (f *FakeCmd) SetDir(dir string) {}

func (f *FakeCmd) SetStdin(in io.Reader) {}

func (f *FakeCmd) SetStdout(out io.Writer) {
	f.writer = out
}

func (f *FakeCmd) SetStderr(out io.Writer) {
	f.writer = out
}

func (f *FakeCmd) SetEnv(env []string) {}

func (f *FakeCmd) Stop() {}

func (f *FakeCmd) Start() error {
	if f.writer != nil {
		f.writer.Write(f.out)
		return f.err
	}
	return f.err
}

func (f *FakeCmd) Wait() error { return nil }

func (f *FakeCmd) StdoutPipe() (io.ReadCloser, error) {
	return nil, nil
}

func (f *FakeCmd) StderrPipe() (io.ReadCloser, error) {
	return nil, nil
}

func TestExec(t *testing.T) {
	prober := New()

	tenKilobyte := strings.Repeat("logs-123", 128*10)      // 8*128*10=10240 = 10KB of text.
	elevenKilobyte := strings.Repeat("logs-123", 8*128*11) // 8*128*11=11264 = 11KB of text.
	fakeExecRunOutput := "fake output (combined stdout and stderr, this content depends on the output of user script)"
	fakeProcessHaltedMsg := "command terminated with exit code 1"

	cases := []struct {
		name           string
		expectedStatus probe.Result
		expectOutput   string
		expectGetError bool
		cmdError       error
		cmdOutput      string
	}{
		{
			name:           "exec probe success",
			expectedStatus: probe.Success,
			expectOutput:   fakeExecRunOutput,
			cmdOutput:      fakeExecRunOutput,
		},
		{
			name:           "exec probe success with truncated output",
			expectedStatus: probe.Success,
			expectOutput:   tenKilobyte,
			cmdOutput:      elevenKilobyte,
		},
		{
			name:           "exec probe success with 0 exit code",
			expectedStatus: probe.Success,
			expectOutput:   fakeExecRunOutput,
			cmdError:       exec.NewCodeExitError(errors.New("no error"), 0),
			cmdOutput:      fakeExecRunOutput,
		},
		{
			name:           "exec probe failed with non-0 exit code",
			expectedStatus: probe.Failure,
			expectOutput:   "fake output (combined stdout and stderr, this content depends on the output of user script)\ncommand terminated with exit code 1",
			cmdError:       exec.NewCodeExitError(errors.New(fakeProcessHaltedMsg), 1),
			cmdOutput:      fakeExecRunOutput,
		},
		{
			name:           "exec probe failed with timeout",
			expectedStatus: probe.Failure,
			expectOutput:   "command testcmd timed out",
			cmdError:       NewTimeoutError(errors.New("command testcmd timed out"), time.Second),
		},
		{
			name:           "unknown exec probe result",
			expectedStatus: probe.Unknown,
			cmdError:       errors.New("unexpected error(not CodeExitError & TimeoutError)"),
			expectGetError: true,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fake := FakeCmd{
				out: []byte(c.cmdOutput),
				err: c.cmdError,
			}
			status, output, err := prober.Probe(&fake)

			assert.Equal(t, c.expectedStatus, status)
			assert.Equal(t, c.expectOutput, output)
			if c.expectGetError {
				assert.Error(t, err)
			} else {
				assert.Nil(t, err)
			}
		})
	}
}
