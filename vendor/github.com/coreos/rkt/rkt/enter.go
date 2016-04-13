// Copyright 2014 The rkt Authors
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
	"errors"
	"fmt"
	"io/ioutil"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/stage0"
	"github.com/coreos/rkt/store"
	"github.com/hashicorp/errwrap"
	"github.com/spf13/cobra"
)

var (
	cmdEnter = &cobra.Command{
		Use:   "enter [--app=APPNAME] UUID [CMD [ARGS ...]]",
		Short: "Enter the namespaces of an app within a rkt pod",

		Long: `UUID should be the UUID of a running pod.

By default the CMD that is run is /bin/bash, providing the user with shell
access to the running pod.`,
		Run: ensureSuperuser(runWrapper(runEnter)),
	}
	flagAppName string
)

const (
	defaultCmd = "/bin/bash"
)

func init() {
	cmdRkt.AddCommand(cmdEnter)
	cmdEnter.Flags().StringVar(&flagAppName, "app", "", "name of the app to enter within the specified pod")

	// Disable interspersed flags to stop parsing after the first non flag
	// argument. This is need to permit to correctly handle
	// multiple "IMAGE -- imageargs ---"  options
	cmdEnter.Flags().SetInterspersed(false)
}

func runEnter(cmd *cobra.Command, args []string) (exit int) {
	if len(args) < 1 {
		cmd.Usage()
		return 1
	}

	p, err := getPodFromUUIDString(args[0])
	if err != nil {
		stderr.PrintE("problem retrieving pod", err)
		return 1
	}
	defer p.Close()

	if !p.isRunning() {
		stderr.Printf("pod %q isn't currently running", p.uuid)
		return 1
	}

	podPID, err := p.getContainerPID1()
	if err != nil {
		stderr.PrintE(fmt.Sprintf("unable to determine the pid for pod %q", p.uuid), err)
		return 1
	}

	appName, err := getAppName(p)
	if err != nil {
		stderr.PrintE("unable to determine app name", err)
		return 1
	}

	argv, err := getEnterArgv(p, args)
	if err != nil {
		stderr.PrintE("enter failed", err)
		return 1
	}

	s, err := store.NewStore(getDataDir())
	if err != nil {
		stderr.PrintE("cannot open store", err)
		return 1
	}

	stage1TreeStoreID, err := p.getStage1TreeStoreID()
	if err != nil {
		stderr.PrintE("error getting stage1 treeStoreID", err)
		return 1
	}

	stage1RootFS := s.GetTreeStoreRootFS(stage1TreeStoreID)

	if err = stage0.Enter(p.path(), podPID, *appName, stage1RootFS, argv); err != nil {
		stderr.PrintE("enter failed", err)
		return 1
	}
	// not reached when stage0.Enter execs /enter
	return 0
}

// getAppName returns the app name to enter
// If one was supplied in the flags then it's simply returned
// If the PM contains a single app, that app's name is returned
// If the PM has multiple apps, the names are printed and an error is returned
func getAppName(p *pod) (*types.ACName, error) {
	if flagAppName != "" {
		return types.NewACName(flagAppName)
	}

	// figure out the app name, or show a list if multiple are present
	b, err := ioutil.ReadFile(common.PodManifestPath(p.path()))
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error reading pod manifest"), err)
	}

	m := schema.PodManifest{}
	if err = m.UnmarshalJSON(b); err != nil {
		return nil, errwrap.Wrap(errors.New("invalid pod manifest"), err)
	}

	switch len(m.Apps) {
	case 0:
		return nil, fmt.Errorf("pod contains zero apps")
	case 1:
		return &m.Apps[0].Name, nil
	default:
	}

	stderr.Print("pod contains multiple apps:")
	for _, ra := range m.Apps {
		stderr.Printf("\t%v", ra.Name)
	}

	return nil, fmt.Errorf("specify app using \"rkt enter --app= ...\"")
}

// getEnterArgv returns the argv to use for entering the pod
func getEnterArgv(p *pod, cmdArgs []string) ([]string, error) {
	var argv []string
	if len(cmdArgs) < 2 {
		stderr.Printf("no command specified, assuming %q", defaultCmd)
		argv = []string{defaultCmd}
	} else {
		argv = cmdArgs[1:]
	}

	return argv, nil
}
