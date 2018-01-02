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
	"fmt"
	"strings"

	"github.com/coreos/rkt/common"
	pkgPod "github.com/coreos/rkt/pkg/pod"
	"github.com/coreos/rkt/stage0"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/coreos/rkt/store/treestore"
	"github.com/spf13/cobra"
)

var (
	cmdAttach = &cobra.Command{
		Use:   "attach [--app=APPNAME] [--mode=MODE] UUID",
		Short: "Attach to an app running within a rkt pod",

		Long: `UUID should be the UUID of a running pod.`,
		Run:  ensureSuperuser(runWrapper(runAttach)),
	}
	flagAttachMode string
)

func init() {
	if common.IsExperimentEnabled("attach") {
		cmdRkt.AddCommand(cmdAttach)
		cmdAttach.Flags().StringVar(&flagAppName, "app", "", "name of the app to enter within the specified pod")
		cmdAttach.Flags().StringVar(&flagAttachMode, "mode", "auto", "attach mode")
	}
}

func runAttach(cmd *cobra.Command, args []string) (exit int) {
	if len(args) < 1 {
		cmd.Usage()
		return 254
	}

	uuid := args[0]
	p, err := pkgPod.PodFromUUIDString(getDataDir(), uuid)
	if err != nil {
		stderr.PrintE("problem retrieving pod", err)
		return 254
	}
	defer p.Close()

	if p.State() != pkgPod.Running {
		stderr.Printf("pod %q isn't currently running", p.UUID)
		return 254
	}

	podPID, err := p.ContainerPid1()
	if err != nil {
		stderr.PrintE(fmt.Sprintf("unable to determine the pid for pod %q", p.UUID), err)
		return 254
	}

	appName, err := getAppName(p)
	if err != nil {
		stderr.PrintE("unable to determine app name", err)
		return 254
	}

	s, err := imagestore.NewStore(storeDir())
	if err != nil {
		stderr.PrintE("cannot open store", err)
		return 254
	}

	ts, err := treestore.NewStore(treeStoreDir(), s)
	if err != nil {
		stderr.PrintE("cannot open store", err)
		return 254
	}

	stage1TreeStoreID, err := p.GetStage1TreeStoreID()
	if err != nil {
		stderr.PrintE("error getting stage1 treeStoreID", err)
		return 254
	}

	// prepare stage1/attach flags
	stage1RootFS := ts.GetRootFS(stage1TreeStoreID)
	attachArgs, err := createStage1AttachFlags(flagAttachMode)
	if err != nil {
		stderr.PrintE("invalid attach mode", err)
		return 254
	}
	attachArgs = append(attachArgs,
		fmt.Sprintf("--app=%s", appName),
		fmt.Sprintf("--debug=%t", globalFlags.Debug),
	)

	if err = stage0.Attach(p.Path(), podPID, *appName, stage1RootFS, uuid, attachArgs); err != nil {
		stderr.PrintE("attach failed", err)
		return 254
	}
	// not reached when stage0.Attach execs /enter
	return 0
}

// createStage1AttachFlags parses an attach stage0 CLI "--mode" flag and
// returns options suited for stage1/attach entrypoint invocation
func createStage1AttachFlags(attachMode string) ([]string, error) {
	attachArgs := []string{}

	// list mode: just print endpoints
	if attachMode == "list" {
		attachArgs = append(attachArgs, "--action=list")
		return attachArgs, nil
	}

	// auto-attach mode: stage1-attach will figure out endpoints
	if attachMode == "auto" || attachMode == "" {
		attachArgs = append(attachArgs, "--action=auto-attach")
		return attachArgs, nil
	}

	// custom-attach mode: user specified endpoints
	var customEndpoints struct {
		TTYIn  bool
		TTYOut bool
		Stdin  bool
		Stdout bool
		Stderr bool
	}
	attachArgs = append(attachArgs, "--action=custom-attach")

	// parse comma-separated endpoints for custom attach
	eps := strings.Split(attachMode, ",")
	for _, e := range eps {
		switch e {
		case "stdin":
			customEndpoints.Stdin = true
		case "stdout":
			customEndpoints.Stdout = true
		case "stderr":
			customEndpoints.Stderr = true
		case "tty":
			customEndpoints.TTYIn = true
			customEndpoints.TTYOut = true
		case "tty-in":
			customEndpoints.TTYIn = true
		case "tty-out":
			customEndpoints.TTYOut = true
		default:
			return nil, fmt.Errorf("unknown endpoint %q", e)
		}
	}

	// check that the resulting attach mode is sane
	if !(customEndpoints.TTYIn || customEndpoints.TTYOut || customEndpoints.Stdin || customEndpoints.Stdout || customEndpoints.Stderr) {
		return nil, fmt.Errorf("mode must specify at least one endpoint to attach")
	}
	if (customEndpoints.TTYIn || customEndpoints.TTYOut) && (customEndpoints.Stdin || customEndpoints.Stdout || customEndpoints.Stderr) {
		return nil, fmt.Errorf("incompatible endpoints %q, cannot simultaneously attach TTY and streams", attachMode)
	}

	attachArgs = append(attachArgs,
		fmt.Sprintf("--tty-in=%t", customEndpoints.TTYIn),
		fmt.Sprintf("--tty-out=%t", customEndpoints.TTYOut),
		fmt.Sprintf("--stdin=%t", customEndpoints.Stdin),
		fmt.Sprintf("--stdout=%t", customEndpoints.Stdout),
		fmt.Sprintf("--stderr=%t", customEndpoints.Stderr),
	)
	return attachArgs, nil

}
