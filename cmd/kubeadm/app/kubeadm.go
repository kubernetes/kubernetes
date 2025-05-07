/*
Copyright 2016 The Kubernetes Authors.

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

package app

import (
	"flag"
	"os"

	"github.com/spf13/pflag"

	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/klog/v2"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd"
)

// Run creates and executes new kubeadm command
func Run() error {
	var allFlags flag.FlagSet
	klog.InitFlags(&allFlags)
	// only add the flags that are still supported for kubeadm
	allFlags.VisitAll(func(f *flag.Flag) {
		switch f.Name {
		// kubeadm only exposes the klog flags covered in https://features.k8s.io/2845
		case "v", "vmodule":
			flag.CommandLine.Var(f.Value, f.Name, f.Usage)
		}
	})

	pflag.CommandLine.SetNormalizeFunc(cliflag.WordSepNormalizeFunc)
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)

	cmd := cmd.NewKubeadmCommand(os.Stdin, os.Stdout, os.Stderr)
	return cmd.Execute()
}
