// Copyright 2015 The rkt Authors
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

// inspired by github.com/docker/docker/pkg/reexec

//+build linux

package multicall

import (
	"fmt"
	"os"
	"os/exec"
	"path"
	"syscall"
)

var exePath string

func init() {
	// save the program path
	var err error
	exePath, err = os.Readlink("/proc/self/exe")
	if err != nil {
		panic("cannot get current executable")
	}
}

type commandFn func() error

var commands = make(map[string]commandFn)

// Entrypoint provides the access to a multicall command.
type Entrypoint string

// Add adds a new multicall command. name is the command name and fn is the
// function that will be executed for the specified command. It returns the
// related Entrypoint.
// Packages adding new multicall commands should call Add in their init
// function.
func Add(name string, fn commandFn) Entrypoint {
	if _, ok := commands[name]; ok {
		panic(fmt.Errorf("command with name %q already exists", name))
	}
	commands[name] = fn
	return Entrypoint(name)
}

// MaybeExec should be called at the start of the program, if the process argv[0] is
// a name registered with multicall, the related function will be executed.
// If the functions returns an error, it will be printed to stderr and will
// exit with an exit status of 1, otherwise it will exit with a 0 exit
// status.
func MaybeExec() {
	name := path.Base(os.Args[0])
	if fn, ok := commands[name]; ok {
		if err := fn(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(254)
		}
		os.Exit(0)
	}
}

// Cmd will prepare the *exec.Cmd for the given entrypoint, configured with the
// provided args.
func (e Entrypoint) Cmd(args ...string) *exec.Cmd {
	// append the Entrypoint as argv[0]
	args = append([]string{string(e)}, args...)
	return &exec.Cmd{
		Path: exePath,
		Args: args,
		SysProcAttr: &syscall.SysProcAttr{
			Pdeathsig: syscall.SIGTERM,
		},
	}
}
