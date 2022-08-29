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

package editor

import (
	"io"
	"os"
	"runtime"
	"strings"
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
)

// Editor holds the command-line args to fire up the editor
type Editor struct {
	Args  []string
	Shell bool
}

// NewDefaultEditor creates a struct Editor that uses the OS environment to
// locate the editor program, looking at EDITOR environment variable to find
// the proper command line. If the provided editor has no spaces, or no quotes,
// it is treated as a bare command to be loaded. Otherwise, the string will
// be passed to the user's shell for execution.
func NewDefaultEditor(envs []string) Editor {
	args, shell := defaultEnvEditor(envs)
	return Editor{
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

func defaultEnvEditor(envs []string) ([]string, bool) {
	var editor string
	for _, env := range envs {
		if len(env) > 0 {
			editor = os.Getenv(env)
		}
		if len(editor) > 0 {
			break
		}
	}
	if len(editor) == 0 {
		editor = platformize(defaultEditor, windowsEditor)
	}
	if !strings.Contains(editor, " ") {
		return []string{editor}, false
	}
	if !strings.ContainsAny(editor, "\"'\\") {
		return strings.Split(editor, " "), false
	}
	// rather than parse the shell arguments ourselves, punt to the shell
	shell := defaultEnvShell()
	return append(shell, editor), true
}

// LaunchTempFile reads the provided stream into a temporary file in the given directory
// and file prefix, and then invokes Launch with the path of that file. It will return
// the contents of the file after launch, any errors that occur, and the path of the
// temporary file so the caller can clean it up as needed.
func (e Editor) LaunchTempFile(prefix, suffix string, r io.Reader) ([]byte, string, error) {
	f, err := os.CreateTemp("", prefix+"*"+suffix)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()
	path := f.Name()
	if _, err := io.Copy(f, r); err != nil {
		os.Remove(path)
		return nil, path, err
	}
	// This file descriptor needs to close so the next process (Launch) can claim it.
	f.Close()
	if err := e.Launch(path); err != nil {
		return nil, path, err
	}
	bytes, err := os.ReadFile(path)
	return bytes, path, err
}

func platformize(linux, windows string) string {
	if runtime.GOOS == "windows" {
		return windows
	}
	return linux
}
