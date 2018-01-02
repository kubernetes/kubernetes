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
	"io/ioutil"
	"os"
	"os/exec"

	rktlog "github.com/coreos/rkt/pkg/log"
	stage1common "github.com/coreos/rkt/stage1/common"
	stage1initcommon "github.com/coreos/rkt/stage1/init/common"

	"github.com/appc/spec/schema/types"
)

var (
	flagApp string
	debug   bool
	log     *rktlog.Logger
	diag    *rktlog.Logger
)

func init() {
	flag.StringVar(&flagApp, "app", "", "Application name")
	flag.BoolVar(&debug, "debug", false, "Run in debug mode")
}

func main() {
	flag.Parse()

	stage1initcommon.InitDebug(debug)

	log, diag, _ = rktlog.NewLogSet("app-start", debug)
	if !debug {
		diag.SetOutput(ioutil.Discard)
	}

	appName, err := types.NewACName(flagApp)
	if err != nil {
		log.FatalE("invalid app name", err)
	}

	enterCmd := stage1common.PrepareEnterCmd(false)

	args := enterCmd
	args = append(args, "/usr/bin/systemctl")
	args = append(args, "start")
	args = append(args, appName.String())

	cmd := exec.Cmd{
		Path: args[0],
		Args: args,
	}

	if out, err := cmd.CombinedOutput(); err != nil {
		log.Fatalf("%q failed to start:\n%s", appName, out)
	}

	os.Exit(0)
}
