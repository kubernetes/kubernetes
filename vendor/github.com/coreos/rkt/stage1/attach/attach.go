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

package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"

	"github.com/appc/spec/schema/types"
	rktlog "github.com/coreos/rkt/pkg/log"
	stage1common "github.com/coreos/rkt/stage1/common"
	stage1initcommon "github.com/coreos/rkt/stage1/init/common"
)

var (
	log  *rktlog.Logger
	diag *rktlog.Logger

	app    string
	action string
	debug  bool

	attachTTYIn  bool
	attachTTYOut bool
	attachStdin  bool
	attachStdout bool
	attachStderr bool
)

func init() {
	flag.StringVar(&action, "action", "list", "Action")
	flag.StringVar(&app, "app", "", "Application name")
	flag.BoolVar(&debug, "debug", false, "Run in debug mode")

	flag.BoolVar(&attachTTYIn, "tty-in", false, "attach tty input")
	flag.BoolVar(&attachTTYOut, "tty-out", false, "attach tty output")
	flag.BoolVar(&attachStdin, "stdin", false, "attach stdin")
	flag.BoolVar(&attachStdout, "stdout", false, "attach stdin")
	flag.BoolVar(&attachStderr, "stderr", false, "attach stderr")
}

func main() {
	flag.Parse()

	stage1initcommon.InitDebug(debug)

	log, diag, _ = rktlog.NewLogSet("stage1-attach", debug)
	if !debug {
		diag.SetOutput(ioutil.Discard)
	}

	appName, err := types.NewACName(app)
	if err != nil {
		log.PrintE("invalid application name", err)
		os.Exit(254)
	}

	if action != "list" && action != "auto-attach" && action != "custom-attach" {
		log.Printf("invalid attach action %q", action)
		os.Exit(254)
	}

	args := stage1common.PrepareEnterCmd(false)
	args = append(args,
		"/iottymux",
		fmt.Sprintf("--action=%s", action),
		fmt.Sprintf("--app=%s", appName),
	)

	cmd := exec.Cmd{
		Path:   args[0],
		Args:   args,
		Stdin:  os.Stdin,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
		Env: []string{
			fmt.Sprintf("PATH=%s", os.Getenv("PATH")),
			fmt.Sprintf("STAGE2_DEBUG=%t", debug),
			fmt.Sprintf("STAGE2_ATTACH_TTYIN=%t", attachTTYIn),
			fmt.Sprintf("STAGE2_ATTACH_TTYOUT=%t", attachTTYOut),
			fmt.Sprintf("STAGE2_ATTACH_STDIN=%t", attachStdin),
			fmt.Sprintf("STAGE2_ATTACH_STDOUT=%t", attachStdout),
			fmt.Sprintf("STAGE2_ATTACH_STDERR=%t", attachStderr),
		},
	}

	if err := cmd.Run(); err != nil {
		log.PrintE(`error executing "iottymux"`, err)
		os.Exit(254)
	}

	os.Exit(0)
}
