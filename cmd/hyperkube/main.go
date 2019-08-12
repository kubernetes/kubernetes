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

// A binary that can morph into all of the other kubernetes binaries. You can
// also soft-link to it busybox style.
//
package main

import (
	goflag "flag"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/logs"
	cloudcontrollermanager "k8s.io/kubernetes/cmd/cloud-controller-manager/app"
	kubeapiserver "k8s.io/kubernetes/cmd/kube-apiserver/app"
	kubecontrollermanager "k8s.io/kubernetes/cmd/kube-controller-manager/app"
	kubeproxy "k8s.io/kubernetes/cmd/kube-proxy/app"
	kubescheduler "k8s.io/kubernetes/cmd/kube-scheduler/app"
	kubelet "k8s.io/kubernetes/cmd/kubelet/app"
	_ "k8s.io/kubernetes/pkg/client/metrics/prometheus" // for client metric registration
	kubectl "k8s.io/kubernetes/pkg/kubectl/cmd"
	_ "k8s.io/kubernetes/pkg/version/prometheus" // for version metric registration
)

func main() {
	rand.Seed(time.Now().UnixNano())

	hyperkubeCommand, allCommandFns := NewHyperKubeCommand()

	// TODO: once we switch everything over to Cobra commands, we can go back to calling
	// cliflag.InitFlags() (by removing its pflag.Parse() call). For now, we have to set the
	// normalize func and add the go flag set by hand.
	pflag.CommandLine.SetNormalizeFunc(cliflag.WordSepNormalizeFunc)
	pflag.CommandLine.AddGoFlagSet(goflag.CommandLine)
	// cliflag.InitFlags()
	logs.InitLogs()
	defer logs.FlushLogs()

	basename := filepath.Base(os.Args[0])
	if err := commandFor(basename, hyperkubeCommand, allCommandFns).Execute(); err != nil {
		os.Exit(1)
	}
}

func commandFor(basename string, defaultCommand *cobra.Command, commands []func() *cobra.Command) *cobra.Command {
	for _, commandFn := range commands {
		command := commandFn()
		if command.Name() == basename {
			return command
		}
		for _, alias := range command.Aliases {
			if alias == basename {
				return command
			}
		}
	}

	return defaultCommand
}

// NewHyperKubeCommand is the entry point for hyperkube
func NewHyperKubeCommand() (*cobra.Command, []func() *cobra.Command) {
	// these have to be functions since the command is polymorphic. Cobra wants you to be top level
	// command to get executed
	apiserver := func() *cobra.Command { return kubeapiserver.NewAPIServerCommand() }
	controller := func() *cobra.Command { return kubecontrollermanager.NewControllerManagerCommand() }
	proxy := func() *cobra.Command { return kubeproxy.NewProxyCommand() }
	scheduler := func() *cobra.Command { return kubescheduler.NewSchedulerCommand() }
	kubectlCmd := func() *cobra.Command { return kubectl.NewDefaultKubectlCommand() }
	kubelet := func() *cobra.Command { return kubelet.NewKubeletCommand() }
	cloudController := func() *cobra.Command {
		cmd := cloudcontrollermanager.NewCloudControllerManagerCommand()
		cmd.Deprecated = "please use the cloud controller manager specific " +
			"to your external cloud provider"
		return cmd
	}

	commandFns := []func() *cobra.Command{
		apiserver,
		controller,
		proxy,
		scheduler,
		kubectlCmd,
		kubelet,
		cloudController,
	}

	cmd := &cobra.Command{
		Use:   "hyperkube",
		Short: "Request a new project",
		Run: func(cmd *cobra.Command, args []string) {
			if len(args) != 0 {
				cmd.Help()
				os.Exit(1)
			}

		},
	}

	for i := range commandFns {
		cmd.AddCommand(commandFns[i]())
	}

	return cmd, commandFns
}
