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
	osexec "os/exec"
	"testing"
)

func TestExecutorNoArgs(t *testing.T) {
	ex := New()

	cmd := ex.Command("true")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if len(out) != 0 {
		t.Errorf("expected no output, got %q", string(out))
	}

	cmd = ex.Command("false")
	out, err = cmd.CombinedOutput()
	if err == nil {
		t.Errorf("expected failure, got nil error")
	}
	if len(out) != 0 {
		t.Errorf("expected no output, got %q", string(out))
	}
	ee, ok := err.(ExitError)
	if !ok {
		t.Errorf("expected an ExitError, got %+v", err)
	}
	if ee.Exited() {
		if code := ee.ExitStatus(); code != 1 {
			t.Errorf("expected exit status 1, got %d", code)
		}
	}

	cmd = ex.Command("/does/not/exist")
	out, err = cmd.CombinedOutput()
	if err == nil {
		t.Errorf("expected failure, got nil error")
	}
	if ee, ok := err.(ExitError); ok {
		t.Errorf("expected non-ExitError, got %+v", ee)
	}
}

func TestExecutorWithArgs(t *testing.T) {
	ex := New()

	cmd := ex.Command("/bin/echo", "stdout")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("expected success, got %+v", err)
	}
	if string(out) != "stdout\n" {
		t.Errorf("unexpected output: %q", string(out))
	}

	cmd = ex.Command("/bin/sh", "-c", "echo stderr > /dev/stderr")
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Errorf("expected success, got %+v", err)
	}
	if string(out) != "stderr\n" {
		t.Errorf("unexpected output: %q", string(out))
	}
}

func TestLookPath(t *testing.T) {
	ex := New()

	shExpected, _ := osexec.LookPath("sh")
	sh, _ := ex.LookPath("sh")
	if sh != shExpected {
		t.Errorf("unexpected result for LookPath: got %s, expected %s", sh, shExpected)
	}
}
