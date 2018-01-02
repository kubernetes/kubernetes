// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package activation

import (
	"bytes"
	"io"
	"os"
	"os/exec"
	"testing"
)

// correctStringWritten fails the text if the correct string wasn't written
// to the other side of the pipe.
func correctStringWritten(t *testing.T, r *os.File, expected string) bool {
	bytes := make([]byte, len(expected))
	io.ReadAtLeast(r, bytes, len(expected))

	if string(bytes) != expected {
		t.Fatalf("Unexpected string %s", string(bytes))
	}

	return true
}

// TestActivation forks out a copy of activation.go example and reads back two
// strings from the pipes that are passed in.
func TestActivation(t *testing.T) {
	cmd := exec.Command("go", "run", "../examples/activation/activation.go")

	r1, w1, _ := os.Pipe()
	r2, w2, _ := os.Pipe()
	cmd.ExtraFiles = []*os.File{
		w1,
		w2,
	}

	cmd.Env = os.Environ()
	cmd.Env = append(cmd.Env, "LISTEN_FDS=2", "FIX_LISTEN_PID=1")

	err := cmd.Run()
	if err != nil {
		t.Fatalf(err.Error())
	}

	correctStringWritten(t, r1, "Hello world")
	correctStringWritten(t, r2, "Goodbye world")
}

func TestActivationNoFix(t *testing.T) {
	cmd := exec.Command("go", "run", "../examples/activation/activation.go")
	cmd.Env = os.Environ()
	cmd.Env = append(cmd.Env, "LISTEN_FDS=2")

	out, _ := cmd.CombinedOutput()
	if bytes.Contains(out, []byte("No files")) == false {
		t.Fatalf("Child didn't error out as expected")
	}
}

func TestActivationNoFiles(t *testing.T) {
	cmd := exec.Command("go", "run", "../examples/activation/activation.go")
	cmd.Env = os.Environ()
	cmd.Env = append(cmd.Env, "LISTEN_FDS=0", "FIX_LISTEN_PID=1")

	out, _ := cmd.CombinedOutput()
	if bytes.Contains(out, []byte("No files")) == false {
		t.Fatalf("Child didn't error out as expected")
	}
}
