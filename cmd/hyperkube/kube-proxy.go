/*
Copyright 2015 The Kubernetes Authors.

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
	"flag"

	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/kubernetes/cmd/kube-proxy/app"
)

// NewKubeProxy creates a new hyperkube Server object that includes the
// description and flags.
func NewKubeProxy() *Server {
	healthz.DefaultHealthz()

	command := app.NewProxyCommand()

	hks := Server{
		name:            "proxy",
		AlternativeName: "kube-proxy",
		SimpleUsage:     "proxy",
		Long:            command.Long,
	}

	serverFlags := hks.Flags()
	serverFlags.AddFlagSet(command.Flags())

	// FIXME this is here because hyperkube does its own flag parsing, and we need
	// the command to know about the go flag set. Remove this once hyperkube is
	// refactored to use cobra throughout.
	command.Flags().AddGoFlagSet(flag.CommandLine)

	hks.Run = func(_ *Server, args []string) error {
		command.SetArgs(args)
		return command.Execute()
	}

	return &hks
}
