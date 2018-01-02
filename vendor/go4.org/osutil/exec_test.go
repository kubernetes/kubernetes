// Copyright 2015 The go4 Authors
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

package osutil

import (
	"fmt"
	"os"
	osexec "os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

const executable_EnvVar = "OSTEST_OUTPUT_EXECPATH"

func TestExecutable(t *testing.T) {
	if runtime.GOOS == "nacl" {
		t.Skip()
	}
	ep, err := Executable()
	if err != nil {
		switch goos := runtime.GOOS; goos {
		case "openbsd": // procfs is not mounted by default
			t.Skipf("Executable failed on %s: %v, expected", goos, err)
		}
		t.Fatalf("Executable failed: %v", err)
	}
	// we want fn to be of the form "dir/prog"
	dir := filepath.Dir(filepath.Dir(ep))
	fn, err := filepath.Rel(dir, ep)
	if err != nil {
		t.Fatalf("filepath.Rel: %v", err)
	}
	cmd := &osexec.Cmd{}
	// make child start with a relative program path
	cmd.Dir = dir
	cmd.Path = fn
	// forge argv[0] for child, so that we can verify we could correctly
	// get real path of the executable without influenced by argv[0].
	cmd.Args = []string{"-", "-test.run=XXXX"}
	cmd.Env = []string{fmt.Sprintf("%s=1", executable_EnvVar)}
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("exec(self) failed: %v", err)
	}
	outs := string(out)
	if !filepath.IsAbs(outs) {
		t.Fatalf("Child returned %q, want an absolute path", out)
	}
	if !sameFile(outs, ep) {
		t.Fatalf("Child returned %q, not the same file as %q", out, ep)
	}
}

func sameFile(fn1, fn2 string) bool {
	fi1, err := os.Stat(fn1)
	if err != nil {
		return false
	}
	fi2, err := os.Stat(fn2)
	if err != nil {
		return false
	}
	return os.SameFile(fi1, fi2)
}

func init() {
	if e := os.Getenv(executable_EnvVar); e != "" {
		// first chdir to another path
		dir := "/"
		if runtime.GOOS == "windows" {
			dir = filepath.VolumeName(".")
		}
		os.Chdir(dir)
		if ep, err := Executable(); err != nil {
			fmt.Fprint(os.Stderr, "ERROR: ", err)
		} else {
			fmt.Fprint(os.Stderr, ep)
		}
		os.Exit(0)
	}
}
