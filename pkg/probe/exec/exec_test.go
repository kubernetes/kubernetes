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
	"fmt"
	"io"
	"strings"
	"testing"

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

type fakeExitError struct {
	exited     bool
	statusCode int
}

func (f *fakeExitError) String() string {
	return f.Error()
}

func (f *fakeExitError) Error() string {
	return "fake exit"
}

func (f *fakeExitError) Exited() bool {
	return f.exited
}

func (f *fakeExitError) ExitStatus() int {
	return f.statusCode
}

func TestExec(t *testing.T) {
	prober := New()

	tenKilobyte := strings.Repeat("logs-123", 128*10)      // 8*128*10=10240 = 10KB of text.
	elevenKilobyte := strings.Repeat("logs-123", 8*128*11) // 8*128*11=11264 = 11KB of text.

	tests := []struct {
		expectedStatus probe.Result
		expectError    bool
		input          string
		output         string
		err            error
	}{
		// Ok
		{probe.Success, false, "OK", "OK", nil},
		// Ok
		{probe.Success, false, "OK", "OK", &fakeExitError{true, 0}},
		// Ok - truncated output
		{probe.Success, false, elevenKilobyte, tenKilobyte, nil},
		// Run returns error
		{probe.Unknown, true, "", "", fmt.Errorf("test error")},
		// Unhealthy
		{probe.Failure, false, "Fail", "", &fakeExitError{true, 1}},
	}
	for i, test := range tests {
		fake := FakeCmd{
			out: []byte(test.output),
			err: test.err,
		}
		status, output, err := prober.Probe(&fake)
		if status != test.expectedStatus {
			t.Errorf("[%d] expected %v, got %v", i, test.expectedStatus, status)
		}
		if err != nil && test.expectError == false {
			t.Errorf("[%d] unexpected error: %v", i, err)
		}
		if err == nil && test.expectError == true {
			t.Errorf("[%d] unexpected non-error", i)
		}
		if test.output != output {
			t.Errorf("[%d] expected %s, got %s", i, test.output, output)
		}
	}
}
