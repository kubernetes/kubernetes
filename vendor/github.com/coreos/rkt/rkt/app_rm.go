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

	pkgPod "github.com/coreos/rkt/pkg/pod"
	"github.com/coreos/rkt/stage0"

	"github.com/appc/spec/schema/types"
	"github.com/spf13/cobra"
)

var (
	cmdAppRm = &cobra.Command{
		Use:   "rm UUID --app=NAME",
		Short: "Remove an app from a pod",
		Long:  "This removes an appplication from a mutable pod, stopping it beforehand if still running.",
		Run:   runWrapper(runAppRm),
	}
)

func init() {
	cmdAppRm.Flags().StringVar(&flagAppName, "app", "", "app to remove")
	cmdApp.AddCommand(cmdAppRm)
}

func runAppRm(cmd *cobra.Command, args []string) (exit int) {
	if len(args) < 1 {
		stderr.Print("must provide the pod UUID")
		return 254
	}

	if flagAppName == "" {
		stderr.Print("must provide the app to remove")
		return 254
	}

	p, err := pkgPod.PodFromUUIDString(getDataDir(), args[0])
	if err != nil {
		stderr.PrintE("problem retrieving pod", err)
		return 254
	}
	defer p.Close()

	appName, err := types.NewACName(flagAppName)
	if err != nil {
		stderr.PrintE("invalid app name", err)
	}

	if p.State() != pkgPod.Running {
		stderr.Printf("pod %q is not running", p.UUID)
		return 254
	}

	if !p.IsSupervisorReady() {
		stderr.Printf("supervisor for pod %q is not ready yet", p.UUID)
		return 254
	}

	podPID, err := p.ContainerPid1()
	if err != nil {
		stderr.PrintE(fmt.Sprintf("unable to determine the pid for pod %q", p.UUID), err)
		return 254
	}

	ccfg := stage0.CommonConfig{
		DataDir: getDataDir(),
		UUID:    p.UUID,
		Debug:   globalFlags.Debug,
	}

	cfg := stage0.RmConfig{
		CommonConfig: &ccfg,
		UsesOverlay:  p.UsesOverlay(),
		AppName:      appName,
		PodPath:      p.Path(),
		PodPID:       podPID,
	}

	if globalFlags.Debug {
		stage0.InitDebug()
	}

	err = stage0.RmApp(cfg)
	if err != nil {
		stderr.PrintE("error removing app", err)
		return 254
	}

	return 0
}
