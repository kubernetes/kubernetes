/*
Copyright 2021 The Kubernetes Authors.

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

// typebuilder-gen is a tool for auto-generating apply builder functions.
package main

import (
	"flag"

	"github.com/spf13/pflag"
	"k8s.io/klog/v2"

	generatorargs "k8s.io/code-generator/cmd/applyconfiguration-gen/args"
	"k8s.io/code-generator/cmd/applyconfiguration-gen/generators"
	"k8s.io/code-generator/pkg/util"
)

func main() {
	klog.InitFlags(nil)
	genericArgs, customArgs := generatorargs.NewDefaults()
	genericArgs.GoHeaderFilePath = util.BoilerplatePath()
	genericArgs.AddFlags(pflag.CommandLine)
	customArgs.AddFlags(pflag.CommandLine, "k8s.io/kubernetes/pkg/apis") // TODO: move this input path out of client-gen
	if err := flag.Set("logtostderr", "true"); err != nil {
		klog.Fatalf("Error: %v", err)
	}
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()

	if err := generatorargs.Validate(genericArgs); err != nil {
		klog.Fatalf("Error: %v", err)
	}

	// Run it.
	if err := genericArgs.Execute(
		generators.NameSystems(),
		generators.DefaultNameSystem(),
		generators.Packages,
	); err != nil {
		klog.Fatalf("Error: %v", err)
	}
	klog.V(2).Info("Completed successfully.")
}
