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
	"syscall"
)

var (
	force bool
)

func init() {
	flag.BoolVar(&force, "force", false, "Forced stopping")
}

func readIntFromFile(path string) (i int, err error) {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		return
	}
	_, err = fmt.Sscanf(string(b), "%d", &i)
	return
}

func main() {
	flag.Parse()
	var sig syscall.Signal

	if force {
		sig = syscall.SIGKILL
	} else {
		sig = syscall.SIGTERM
	}

	pid, err := readIntFromFile("pid")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading pid: %v\n", err)
		os.Exit(254)
	}

	// check for process existence, then kill the process otherwise the group
	err = syscall.Kill(pid, syscall.Signal(0))
	if err == nil {
		err = syscall.Kill(pid, sig)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error stopping process %d: %v\n", pid, err)
			os.Exit(254)
		}
	} else {
		pgid, err := readIntFromFile("pgid")
		if err != nil {
			fmt.Fprintf(os.Stderr, "error reading pgid: %v\n", err)
			os.Exit(254)
		}
		err = syscall.Kill(-pgid, sig)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error stopping process group %d: %v\n", pgid, err)
			os.Exit(254)
		}
	}
}
