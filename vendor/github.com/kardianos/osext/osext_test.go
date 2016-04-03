// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin linux freebsd netbsd windows

package osext

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

const (
	executableEnvVar = "OSTEST_OUTPUT_EXECUTABLE"

	executableEnvValueMatch  = "match"
	executableEnvValueDelete = "delete"
)

func TestExecutableMatch(t *testing.T) {
	ep, err := Executable()
	if err != nil {
		t.Fatalf("Executable failed: %v", err)
	}

	// fullpath to be of the form "dir/prog".
	dir := filepath.Dir(filepath.Dir(ep))
	fullpath, err := filepath.Rel(dir, ep)
	if err != nil {
		t.Fatalf("filepath.Rel: %v", err)
	}
	// Make child start with a relative program path.
	// Alter argv[0] for child to verify getting real path without argv[0].
	cmd := &exec.Cmd{
		Dir:  dir,
		Path: fullpath,
		Env:  []string{fmt.Sprintf("%s=%s", executableEnvVar, executableEnvValueMatch)},
	}
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

func TestExecutableDelete(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip()
	}
	fpath, err := Executable()
	if err != nil {
		t.Fatalf("Executable failed: %v", err)
	}

	r, w := io.Pipe()
	stderrBuff := &bytes.Buffer{}
	stdoutBuff := &bytes.Buffer{}
	cmd := &exec.Cmd{
		Path:   fpath,
		Env:    []string{fmt.Sprintf("%s=%s", executableEnvVar, executableEnvValueDelete)},
		Stdin:  r,
		Stderr: stderrBuff,
		Stdout: stdoutBuff,
	}
	err = cmd.Start()
	if err != nil {
		t.Fatalf("exec(self) start failed: %v", err)
	}

	tempPath := fpath + "_copy"
	_ = os.Remove(tempPath)

	err = copyFile(tempPath, fpath)
	if err != nil {
		t.Fatalf("copy file failed: %v", err)
	}
	err = os.Remove(fpath)
	if err != nil {
		t.Fatalf("remove running test file failed: %v", err)
	}
	err = os.Rename(tempPath, fpath)
	if err != nil {
		t.Fatalf("rename copy to previous name failed: %v", err)
	}

	w.Write([]byte{0})
	w.Close()

	err = cmd.Wait()
	if err != nil {
		t.Fatalf("exec wait failed: %v", err)
	}

	childPath := stderrBuff.String()
	if !filepath.IsAbs(childPath) {
		t.Fatalf("Child returned %q, want an absolute path", childPath)
	}
	if !sameFile(childPath, fpath) {
		t.Fatalf("Child returned %q, not the same file as %q", childPath, fpath)
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
func copyFile(dest, src string) error {
	df, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer df.Close()

	sf, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sf.Close()

	_, err = io.Copy(df, sf)
	return err
}

func TestMain(m *testing.M) {
	env := os.Getenv(executableEnvVar)
	switch env {
	case "":
		os.Exit(m.Run())
	case executableEnvValueMatch:
		// First chdir to another path.
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
	case executableEnvValueDelete:
		bb := make([]byte, 1)
		var err error
		n, err := os.Stdin.Read(bb)
		if err != nil {
			fmt.Fprint(os.Stderr, "ERROR: ", err)
			os.Exit(2)
		}
		if n != 1 {
			fmt.Fprint(os.Stderr, "ERROR: n != 1, n == ", n)
			os.Exit(2)
		}
		if ep, err := Executable(); err != nil {
			fmt.Fprint(os.Stderr, "ERROR: ", err)
		} else {
			fmt.Fprint(os.Stderr, ep)
		}
	}
	os.Exit(0)
}
