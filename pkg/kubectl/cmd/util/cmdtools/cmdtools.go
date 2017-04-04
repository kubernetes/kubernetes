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

package cmdtools

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

	"k8s.io/kubernetes/pkg/util/term"
)

const (
	// sorry, blame Git
	// TODO: on Windows rely on 'start' to launch the editor associated
	// with the given file type. If we can't because of the need of
	// blocking, use a script with 'ftype' and 'assoc' to detect it.
	defaultEditor = "vi"
	defaultShell  = "/bin/bash"
	windowsEditor = "notepad"
	windowsShell  = "cmd"

	//for diff command
	gitDiff     = "git diff --no-index"
	defaultDiff = "diff"
)

// CmdTools include
// editor and diff tools
type CmdTool struct {
	Name  string
	Args  []string
	Shell bool
}

// NewDefaultCmdTool creates a struct CmdTool that uses the OS environment to
// locate the editor or diff program, looking at environment variable to find
// the proper command line. for editor: If the provided editor has no spaces,
// or no quotes, it is treated as a bare command to be loaded.
// Otherwise, the string will be passed to the user's shell for execution.
func NewDefaultCmdTool(name string, envs []string) CmdTool {
	args, shell := defaultEnv(name, envs)
	return CmdTool{
		Name:  name,
		Args:  args,
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

func defaultEnv(cmdName string, envs []string) ([]string, bool) {
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
		case "editor":
			cmd = platformize(defaultEditor, windowsEditor)
		case "diff":
			cmd = checkGitDiffInstalled(gitDiff, defaultDiff)
		}
	}

	if !strings.Contains(cmd, " ") {
		return []string{cmd}, false
	}
	if !strings.ContainsAny(cmd, "\"'\\") {
		return strings.Split(cmd, " "), false
	}
	// rather than parse the shell arguments ourselves, punt to the shell
	shell := defaultEnvShell()
	return append(shell, cmd), true
}

func (t CmdTool) args(head []string, path string) []string {
	args := make([]string, len(head))
	copy(args, head)
	if t.Shell {
		last := args[len(args)-1]
		args[len(args)-1] = fmt.Sprintf("%s %q", last, path)
	} else {
		args = append(args, path)
	}
	return args
}

// Launch opens the described or returns an error. The TTY will be protected, and
// SIGQUIT, SIGTERM, and SIGINT will all be trapped.
func (t CmdTool) Launch(paths []string) error {
	args := make([]string, len(t.Args))
	copy(args, t.Args)
	if len(t.Args) == 0 {
		return fmt.Errorf("no %s defined, can't open %s", t.Name, paths[0])
	}
	for _, path := range paths {
		abs, err := filepath.Abs(path)
		if err != nil {
			return err
		}
		args = t.args(args, abs)
	}
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	glog.V(5).Infof("Opening file with %s %v", t.Name, args)
	if err := (term.TTY{In: os.Stdin, TryDev: true}).Safe(cmd.Run); err != nil {
		if err, ok := err.(*exec.Error); ok {
			if err.Err == exec.ErrNotFound {
				return fmt.Errorf("unable to launch the %s %q", t.Name, strings.Join(t.Args, " "))
			}
		}

		//diff command will return 1 when compare different
		if t.Name == "diff" {
			if _, ok := err.(*exec.ExitError); ok {
				return nil
			}
		}

		return fmt.Errorf("there was a problem with the %s %q", t.Name, strings.Join(t.Args, " "))
	}
	return nil
}

// LaunchTempFile reads the provided stream into one or two temporary file in the given directory
// and file prefix, and then invokes Launch with the path of that file. It will return
// the contents of the first file after launch, any errors that occur, and the path of the
// temporary file so the caller can clean it up as needed.
func (t CmdTool) LaunchTempFile(prefix, suffix string, r1, r2 io.Reader) ([]byte, string, error) {
	var paths []string

	path, err := prepareTempFile(prefix, suffix, r1)
	if err != nil {
		return nil, "", err
	}
	paths = append(paths, path)

	//diff cmd should prepare two files
	if t.Name == "diff" {
		path, err = prepareTempFile(prefix, suffix, r2)
		if err != nil {
			return nil, "", err
		}
		paths = append(paths, path)
	}
	if err := t.Launch(paths); err != nil {
		return nil, path, err
	}

	//edit cmd should return the first file's content
	bytes, err := ioutil.ReadFile(paths[0])
	return bytes, path, err
}

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
	cmd := exec.Command("git", "version")
	err := cmd.Run()
	if err != nil {
		return diff
	}
	return gitdiff
}
