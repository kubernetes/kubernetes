/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"

	"github.com/docker/docker/pkg/term"
	"github.com/golang/glog"
)

const (
	// sorry, blame Git
	defaultEditor = "vi"
	defaultShell  = "/bin/bash"
)

type Editor struct {
	Args  []string
	Shell bool
}

// NewDefaultEditor creates a struct Editor that uses the OS environment to
// locate the editor program, looking at EDITOR environment variable to find
// the proper command line. If the provided editor has no spaces, or no quotes,
// it is treated as a bare command to be loaded. Otherwise, the string will
// be passed to the user's shell for execution.
func NewDefaultEditor() Editor {
	args, shell := defaultEnvEditor()
	return Editor{
		Args:  args,
		Shell: shell,
	}
}

func defaultEnvShell() []string {
	shell := os.Getenv("SHELL")
	if len(shell) == 0 {
		shell = defaultShell
	}
	return []string{shell, "-c"}
}

func defaultEnvEditor() ([]string, bool) {
	editor := os.Getenv("EDITOR")
	if len(editor) == 0 {
		editor = defaultEditor
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

func (e Editor) args(path string) []string {
	args := make([]string, len(e.Args))
	copy(args, e.Args)
	if e.Shell {
		last := args[len(args)-1]
		args[len(args)-1] = fmt.Sprintf("%s %q", last, path)
	} else {
		args = append(args, path)
	}
	return args
}

// Launch opens the described or returns an error. The TTY will be protected, and
// SIGQUIT, SIGTERM, and SIGINT will all be trapped.
func (e Editor) Launch(path string) error {
	if len(e.Args) == 0 {
		return fmt.Errorf("no editor defined, can't open %s", path)
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	args := e.args(abs)
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	glog.V(5).Infof("Opening file with editor %v", args)
	if err := withSafeTTYAndInterrupts(cmd.Run); err != nil {
		if err, ok := err.(*exec.Error); ok {
			if err.Err == exec.ErrNotFound {
				return fmt.Errorf("unable to launch the editor %q", strings.Join(e.Args, " "))
			}
		}
		return fmt.Errorf("there was a problem with the editor %q", strings.Join(e.Args, " "))
	}
	return nil
}

// LaunchTempFile reads the provided stream into a temporary file in the given directory
// and file prefix, and then invokes Launch with the path of that file. It will return
// the contents of the file after launch, any errors that occur, and the path of the
// temporary file so the caller can clean it up as needed.
func (e Editor) LaunchTempFile(prefix, suffix string, r io.Reader) ([]byte, string, error) {
	f, err := tempFile(prefix, suffix)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()
	path := f.Name()
	if _, err := io.Copy(f, r); err != nil {
		os.Remove(path)
		return nil, path, err
	}
	if err := e.Launch(path); err != nil {
		return nil, path, err
	}
	bytes, err := ioutil.ReadFile(path)
	return bytes, path, err
}

// withSafeTTYAndInterrupts invokes the provided function after the terminal
// state has been stored, and then on any error or termination attempts to
// restore the terminal state to its prior behavior. It also eats signals
// for the duration of the function.
func withSafeTTYAndInterrupts(fn func() error) error {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, childSignals...)
	defer signal.Stop(ch)

	inFd := os.Stdin.Fd()
	if !term.IsTerminal(inFd) {
		if f, err := os.Open("/dev/tty"); err == nil {
			defer f.Close()
			inFd = f.Fd()
		}
	}

	if term.IsTerminal(inFd) {
		state, err := term.SaveState(inFd)
		if err != nil {
			return err
		}
		go func() {
			if _, ok := <-ch; !ok {
				return
			}
			term.RestoreTerminal(inFd, state)
		}()
		defer term.RestoreTerminal(inFd, state)
		return fn()
	}
	return fn()
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
