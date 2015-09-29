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

package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"syscall"

	"k8s.io/kubernetes/contrib/mesos/pkg/redirfd"
)

func main() {
	nonblock := flag.Bool("n", false, "open file in non-blocking mode")
	changemode := flag.Bool("b", false, "change mode of file after opening it: to non-blocking mode if the -n option was not given, to blocking mode if it was")
	flag.Parse()

	args := flag.Args()
	if len(args) < 4 {
		fmt.Fprintf(os.Stderr, "expected {mode} {fd} {file} instead of: %v\n", args)
		os.Exit(1)
	}

	var mode redirfd.RedirectMode
	switch m := args[0]; m {
	case "r":
		mode = redirfd.Read
	case "w":
		mode = redirfd.Write
	case "u":
		mode = redirfd.Update
	case "a":
		mode = redirfd.Append
	case "c":
		mode = redirfd.AppendExisting
	case "x":
		mode = redirfd.WriteNew
	default:
		fmt.Fprintf(os.Stderr, "unrecognized mode %q\n", mode)
		os.Exit(1)
	}

	fd, err := redirfd.ParseFileDescriptor(args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to parse file descriptor: %v\n", err)
		os.Exit(1)
	}
	file := args[2]

	f, err := mode.Redirect(*nonblock, *changemode, fd, file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "redirect failed: %q, %v\n", args[1], err)
		os.Exit(1)
	}
	var pargs []string
	if len(args) > 4 {
		pargs = args[4:]
	}
	cmd := exec.Command(args[3], pargs...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	switch fd {
	case redirfd.Stdin:
		cmd.Stdin = f
	case redirfd.Stdout:
		cmd.Stdout = f
	case redirfd.Stderr:
		cmd.Stderr = f
	default:
		cmd.ExtraFiles = []*os.File{f}
	}
	defer f.Close()
	if err = cmd.Run(); err != nil {
		exiterr := err.(*exec.ExitError)
		state := exiterr.ProcessState
		if state != nil {
			sys := state.Sys()
			if waitStatus, ok := sys.(syscall.WaitStatus); ok {
				if waitStatus.Signaled() {
					os.Exit(256 + int(waitStatus.Signal()))
				} else {
					os.Exit(waitStatus.ExitStatus())
				}
			}
		}
		os.Exit(3)
	}
}
