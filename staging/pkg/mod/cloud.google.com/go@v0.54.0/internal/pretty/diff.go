// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pretty

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"syscall"
)

// Diff compares the pretty-printed representation of two values. The second
// return value reports whether the two values' representations are identical.
// If it is false, the first return value contains the diffs.
//
// The output labels the first value "want" and the second "got".
//
// Diff works by invoking the "diff" command. It will only succeed in
// environments where "diff" is on the shell path.
func Diff(want, got interface{}) (string, bool, error) {
	fname1, err := writeToTemp(want)
	if err != nil {
		return "", false, err
	}
	defer os.Remove(fname1)

	fname2, err := writeToTemp(got)
	if err != nil {
		return "", false, err
	}
	defer os.Remove(fname2)

	cmd := exec.Command("diff", "-u", "--label=want", "--label=got", fname1, fname2)
	out, err := cmd.Output()
	if err == nil {
		return string(out), true, nil
	}
	eerr, ok := err.(*exec.ExitError)
	if !ok {
		return "", false, err
	}
	ws, ok := eerr.Sys().(syscall.WaitStatus)
	if !ok {
		return "", false, err
	}
	if ws.ExitStatus() != 1 {
		return "", false, err
	}
	// Exit status of 1 means no error, but diffs were found.
	return string(out), false, nil
}

func writeToTemp(v interface{}) (string, error) {
	f, err := ioutil.TempFile("", "prettyDiff")
	if err != nil {
		return "", err
	}
	if _, err := fmt.Fprintf(f, "%+v\n", Value(v)); err != nil {
		return "", err
	}
	if err := f.Close(); err != nil {
		return "", err
	}
	return f.Name(), nil
}
