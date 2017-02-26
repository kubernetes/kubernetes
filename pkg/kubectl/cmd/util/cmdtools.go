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

package util

import (
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/kubectl/util/term"
)

const (
	// TODO: on Windows rely on 'start' to launch the editor associated
	// with the given file type. If we can't because of the need of
	// blocking, use a script with 'ftype' and 'assoc' to detect it.
	defaultEditor = "vi"
	defaultShell  = "/bin/bash"
	windowsEditor = "notepad"
	windowsShell  = "cmd"
	gitDiff       = "git diff --no-index"
	normalDiff    = "diff"
)

type CmdName string

const (
	EditorCmd CmdName = "editor"
	DiffCmd   CmdName = "diff"
)

// CmdTools include
// editor and diff tools
type CmdTool struct {
	Name  CmdName
	Args  []string
	Shell bool
}

// NewDefaultCmdTool creates a struct CmdTool that uses the OS environment to
// locate the editor or diff program, looking at environment variable to find
// the proper command line. for editor: If the provided editor has no spaces,
// or no quotes, it is treated as a bare command to be loaded.
// Otherwise, the string will be passed to the user's shell for execution.
func NewDefaultCmdTool(name CmdName, envs []string) CmdTool {
	exec, shell := defaultEnv(name, envs)
	return CmdTool{
		Name:  name,
		Args:  exec,
		Shell: shell,
	}
}

func defaultEnvShell() []string {
	shell := os.Getenv("SHELL")
	if len(shell) == 0 {
		shell = platformize(defaultShell, windowsShell)
	}
	flag := "-c"
	if shell == windowsShell {
		flag = "/C"
	}
	return []string{shell, flag}
}

func defaultEnv(cmdName CmdName, envs []string) ([]string, bool) {
	var cmd string
	for _, env := range envs {
		if len(env) > 0 {
			cmd = os.Getenv(env)
		}
		if len(cmd) > 0 {
			break
		}
	}
	if len(cmd) == 0 {
		switch cmdName {
		case EditorCmd:
			cmd = platformize(defaultEditor, windowsEditor)
		case DiffCmd:
			cmd = checkGitDiffInstalled(gitDiff, normalDiff)
		}
	}

	if !strings.Contains(cmd, " ") {
		return []string{cmd}, false
	}
	if !strings.ContainsAny(cmd, `"'\`) {
		return strings.Split(cmd, " "), false
	}
	// rather than parse the shell arguments ourselves, punt to the shell
	shell := defaultEnvShell()
	return append(shell, cmd), true
}

func (t CmdTool) args(path string) []string {
	if t.Shell {
		last := t.Args[len(t.Args)-1]
		t.Args[len(t.Args)-1] = fmt.Sprintf("%s %q", last, path)
	} else {
		t.Args = append(t.Args, path)
	}
	return t.Args
}

// Launch opens the described or returns an error. The TTY will be protected, and
// SIGQUIT, SIGTERM, and SIGINT will all be trapped.
func (t CmdTool) Launch(paths []string) error {
	cmdPrefix := make([]string, len(t.Args))
	copy(cmdPrefix, t.Args)
	if len(t.Args) == 0 {
		return fmt.Errorf("no %s defined, can't open %s", t.Name, paths[0])
	}
	for _, path := range paths {
		abs, err := filepath.Abs(path)
		if err != nil {
			return err
		}
		t.Args = t.args(abs)
	}
	cmd := exec.Command(t.Args[0], t.Args[1:]...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	glog.V(5).Infof("Opening file with %s %v", t.Name, t.Args)
	if err := (term.TTY{In: os.Stdin, TryDev: true}).Safe(cmd.Run); err != nil {
		if err, ok := err.(*exec.Error); ok {
			if err.Err == exec.ErrNotFound {
				return fmt.Errorf("unable to launch the %s %q", t.Name, strings.Join(cmdPrefix, " "))
			}
		}

		//diff command will return 1 when compare different
		if t.Name == "diff" {
			if _, ok := err.(*exec.ExitError); ok {
				return nil
			}
		}

		return fmt.Errorf("unknown error with the %s %q", t.Name, strings.Join(cmdPrefix, " "))
	}
	return nil
}

type TempFile struct {
	Prefix string
	Suffix string
	Buffer io.Reader
}

// LaunchTempFiles reads the provided stream into multiple temporary file in the given directory
// and file prefix, and then invokes Launch with the path of that file. It will return
// the contents of the first file after launch, any errors that occur, and the path of the
// temporary file so the caller can clean it up as needed.
func (t CmdTool) LaunchTempFiles(tmpFile ...TempFile) ([]string, error) {
	var paths []string
	for _, f := range tmpFile {
		path, err := prepareTempFile(f.Prefix, f.Suffix, f.Buffer)
		if err != nil {
			return nil, err
		}
		paths = append(paths, path)
	}
	return paths, t.Launch(paths)
}

// LaunchTempFile reads the provided stream into a temporary file in the given directory
// and file prefix, and then invokes LaunchTempFiles return the path of that file. It will return
// the contents of the file after launch, any errors that occur, and the path of the
// temporary file so the caller can clean it up as needed.
func (t CmdTool) LaunchTempFile(tmpFile TempFile) ([]byte, string, error) {
	paths, err := t.LaunchTempFiles(tmpFile)
	if err != nil {
		return nil, "", err
	}
	bytes, err := ioutil.ReadFile(paths[0])
	return bytes, paths[0], err
}

// prepareTempFile create a generated temp file and write stream content into this file
func prepareTempFile(prefix, suffix string, r io.Reader) (string, error) {
	f, err := tempFile(prefix, suffix)
	if err != nil {
		return "", err
	}
	defer f.Close()
	path := f.Name()
	if _, err := io.Copy(f, r); err != nil {
		os.Remove(path)
		return "", err
	}
	// This file descriptor needs to close so the next process (Launch) can claim it.
	f.Close()
	return path, nil
}

// tempFile is a reinvent the wheel of ioutil.TempFile. because ioutil.TempFile does not support extensions for temp files,
// which means temporary files on platforms with extensions do not perform properly. see https://github.com/openshift/origin/pull/1795
func tempFile(prefix, suffix string) (f *os.File, err error) {
	dir := os.TempDir()

	for i := 0; i < 10000; i++ {
		name := filepath.Join(dir, prefix+randSeq(5)+suffix)
		f, err = os.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0600)
		if os.IsExist(err) {
			continue
		}
		break
	}
	return
}

var letters = []rune("abcdefghijklmnopqrstuvwxyz0123456789")

func randSeq(n int) string {
	b := make([]rune, n)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return string(b)
}

func platformize(linux, windows string) string {
	if runtime.GOOS == "windows" {
		return windows
	}
	return linux
}

func checkGitDiffInstalled(gitdiff, diff string) string {
	cmdName := "command"
	args := []string{"-v", "git"}
	if runtime.GOOS == "windows" {
		cmdName = "where"
		args = []string{"git"}
	}
	cmd := exec.Command(cmdName, args...)
	err := cmd.Run()
	if err != nil {
		return diff
	}
	return gitdiff
}
