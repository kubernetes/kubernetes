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
	cmdAppStop = &cobra.Command{
		Use:   "stop UUID --app=NAME",
		Short: "Stop an app in a pod",
		Long:  "This stops an application running inside a mutable pod.",
		Run:   runWrapper(runAppStop),
	}
)

func init() {
	cmdAppStop.Flags().StringVar(&flagAppName, "app", "", "app to stop")
	cmdApp.AddCommand(cmdAppStop)
}

func runAppStop(cmd *cobra.Command, args []string) (exit int) {
	if len(args) < 1 {
		stderr.Print("must provide the pod UUID")
		return 254
	}

	if flagAppName == "" {
		stderr.Print("must provide the app to remove")
		return 254
	}

	appName, err := types.NewACName(flagAppName)
	if err != nil {
		stderr.PrintE("invalid app name", err)
		return 254
	}

	p, err := pkgPod.PodFromUUIDString(getDataDir(), args[0])
	if err != nil {
		stderr.PrintE("problem retrieving pod", err)
		return 254
	}
	defer p.Close()

	if p.IsAfterRun() {
		stdout.Printf("pod %q is already stopped", p.UUID)
		return 0
	}

	if p.State() != pkgPod.Running {
		stderr.Printf("pod %q isn't currently running", p.UUID)
		return 254
	}

	if !p.IsSupervisorReady() {
		stderr.Printf("supervisor for pod %q is not yet ready", p.UUID)
		return 254
	}

	podPID, err := p.ContainerPid1()
	if err != nil {
		stderr.PrintE(fmt.Sprintf("unable to determine the pid for pod %q", p.UUID), err)
		return 254
	}

	cfg := stage0.CommonConfig{
		DataDir: getDataDir(),
		UUID:    p.UUID,
		Debug:   globalFlags.Debug,
	}

	scfg := stage0.StopConfig{
		CommonConfig: &cfg,
		PodPath:      p.Path(),
		AppName:      appName,
		PodPID:       podPID,
	}

	if globalFlags.Debug {
		stage0.InitDebug()
	}

	err = stage0.StopApp(scfg)
	if err != nil {
		stderr.PrintE("error stopping app", err)
		return 254
	}

	return 0
}
