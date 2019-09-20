/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	goflag "flag"
	"math/rand"
	"os"
	"time"

	"github.com/spf13/pflag"

	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/kubectl/pkg/util/logs"
	"k8s.io/kubernetes/cmd/kubectl-sdk/pkg/dispatcher"
	"k8s.io/kubectl/pkg/cmd"

	// Import to initialize client auth plugins.
	"k8s.io/client-go/pkg/version"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	command := cmd.NewDefaultKubectlCommand()

	// TODO: once we switch everything over to Cobra commands, we can go back to calling
	// cliflag.InitFlags() (by removing its pflag.Parse() call). For now, we have to set the
	// normalize func and add the go flag set by hand.
	pflag.CommandLine.SetNormalizeFunc(cliflag.WordSepNormalizeFunc)
	pflag.CommandLine.AddGoFlagSet(goflag.CommandLine)
	// cliflag.InitFlags()
	logs.InitLogs()
	defer logs.FlushLogs()

	// Checks the server version, and executes the matching kubectl version. The convention
	// for the versioned kubectl binary is: kubectl.<MAJOR>.<MINOR>. Example: kubectl.1.12
	// This versioned kubectl binary MUST be in the same directory as this kubectl binary.
	// This does NOT return if it successfully delegates to another version of kubectl,
	// since the current process is overwritten (see execve(2)).
	dispatcher.Execute(version.Get())

	if err := command.Execute(); err != nil {
		os.Exit(1)
	}
}
