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

	rktlog "github.com/coreos/rkt/pkg/log"
	"github.com/coreos/rkt/stage1/common/ssh"
)

var (
	force bool
	log   *rktlog.Logger
)

func init() {
	flag.BoolVar(&force, "force", false, "Forced stop")
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

	if force {
		pid, err := readIntFromFile("pid")
		if err != nil {
			fmt.Fprintf(os.Stderr, "error reading pid: %v\n", err)
			os.Exit(254)
		}

		if err := syscall.Kill(pid, syscall.SIGKILL); err != nil {
			fmt.Fprintf(os.Stderr, "error sending %v: %v\n", syscall.SIGKILL, err)
			os.Exit(254)
		}
		return
	}

	// ExecSSH() should return only with error
	log.Error(ssh.ExecSSH([]string{"systemctl", "halt"}))
	os.Exit(254)
}
