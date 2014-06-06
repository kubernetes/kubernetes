// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin linux freebsd netbsd windows

package osext

import (
	"fmt"
	"os"
	oexec "os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

const execPath_EnvVar = "OSTEST_OUTPUT_EXECPATH"

func TestExecPath(t *testing.T) {
	ep, err := Executable()
	if err != nil {
		t.Fatalf("ExecPath failed: %v", err)
	}
	// we want fn to be of the form "dir/prog"
	dir := filepath.Dir(filepath.Dir(ep))
	fn, err := filepath.Rel(dir, ep)
	if err != nil {
		t.Fatalf("filepath.Rel: %v", err)
	}
	cmd := &oexec.Cmd{}
	// make child start with a relative program path
	cmd.Dir = dir
	cmd.Path = fn
	// forge argv[0] for child, so that we can verify we could correctly
	// get real path of the executable without influenced by argv[0].
	cmd.Args = []string{"-", "-test.run=XXXX"}
	cmd.Env = []string{fmt.Sprintf("%s=1", execPath_EnvVar)}
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
	if e := os.Getenv(execPath_EnvVar); e != "" {
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
