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

package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/user"

	rktlog "github.com/coreos/rkt/pkg/log"
	"github.com/coreos/rkt/stage1/common/ssh"
)

var (
	debug   bool
	podPid  string
	appName string
	u, _    = user.Current()
	log     *rktlog.Logger
	diag    *rktlog.Logger
)

func init() {
	flag.BoolVar(&debug, "debug", false, "Run in debug mode")
	flag.StringVar(&podPid, "pid", "", "podPID")
	flag.StringVar(&appName, "appname", "", "application to use")

	log, diag, _ = rktlog.NewLogSet("kvm", false)
}

func getAppexecArgs() []string {
	// Documentation/devel/stage1-implementors-guide.md#arguments-1
	// also from ../enter/enter.c
	if appName == "" {
		return flag.Args()
	}
	args := []string{
		"/enterexec",
		fmt.Sprintf("/opt/stage2/%s/rootfs", appName),
		"/", // as in ../enter/enter.c - this should be app.WorkingDirectory
		fmt.Sprintf("/rkt/env/%s", appName),
		u.Uid,
		u.Gid,
		"-e", /* entering phase */
		"--",
	}
	return append(args, flag.Args()...)
}

func main() {
	flag.Parse()

	log.SetDebug(debug)
	diag.SetDebug(debug)

	if !debug {
		diag.SetOutput(ioutil.Discard)
	}

	// ExecSSH() should return only with error
	log.Error(ssh.ExecSSH(getAppexecArgs()))

	os.Exit(254)
}
