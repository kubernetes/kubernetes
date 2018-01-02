// Copyright 2016 The rkt Authors
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

//+build linux

package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/shirou/gopsutil/process"
)

var (
	force bool
)

func init() {
	flag.BoolVar(&force, "force", false, "Forced stopping")
}

func readIntFromFile(path string) (i int32, err error) {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		return
	}
	_, err = fmt.Sscanf(string(b), "%d", &i)
	return
}

func main() {
	flag.Parse()

	pid, err := readIntFromFile("ppid")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading pid: %v\n", err)
		os.Exit(254)
	}

	process, err := process.NewProcess(pid)
	if err != nil {
		fmt.Fprintf(os.Stderr, "unable to create process %d instance: %v\n", pid, err)
		os.Exit(254)
	}

	if force {
		children, err := process.Children()
		if err != nil {
			fmt.Fprintf(os.Stderr, "cannot get child processes from %d: %v\n", pid, err)
			os.Exit(254)
		}

		for _, child := range children {
			if err := child.Kill(); err != nil {
				fmt.Fprintf(os.Stderr, "unable to kill process %d: %v\n", child.Pid, err)
				os.Exit(254)
			}
		}

		if err := process.Kill(); err != nil {
			fmt.Fprintf(os.Stderr, "unable to kill process %d: %v\n", pid, err)
			os.Exit(254)
		}
	} else {
		if process.Terminate() != nil {
			fmt.Fprintf(os.Stderr, "unable to terminate process %d: %v\n", pid, err)
			os.Exit(254)
		}
	}
}
