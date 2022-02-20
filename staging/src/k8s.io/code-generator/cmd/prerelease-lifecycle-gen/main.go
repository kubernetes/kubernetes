/*
Copyright 2020 The Kubernetes Authors.

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

// prerelease-lifecycle-gen is a tool for auto-generating api-status.csv files.
//
// Given a list of input directories, it will create a zz_api_status.go file for all beta APIs which indicates the kinds,
// the release it was introduced, the release it will be deprecated, and the release it will be removed.
//
// Generation is governed by comment tags in the source.  Any package may
// request Status generation by including a comment in the file-comments of
// one file, of the form:
//   // +k8s:prerelease-lifecycle-gen=true
//
// // +k8s:prerelease-lifecycle-gen:introduced=1.19
// // +k8s:prerelease-lifecycle-gen:deprecated=1.22
// // +k8s:prerelease-lifecycle-gen:removed=1.25
// // +k8s:prerelease-lifecycle-gen:replacement=wardle.example.com,v1,Flunder
//
// Note that registration is a whole-package option, and is not available for
// individual types.
package main

import (
	"flag"

	"github.com/spf13/pflag"
	generatorargs "k8s.io/code-generator/cmd/prerelease-lifecycle-gen/args"
	statusgenerators "k8s.io/code-generator/cmd/prerelease-lifecycle-gen/prerelease-lifecycle-generators"
	"k8s.io/code-generator/pkg/util"
	"k8s.io/klog/v2"
)

func main() {
	klog.InitFlags(nil)
	genericArgs, customArgs := generatorargs.NewDefaults()

	// Override defaults.
	// TODO: move this out of prerelease-lifecycle-gen
	genericArgs.GoHeaderFilePath = util.BoilerplatePath()

	genericArgs.AddFlags(pflag.CommandLine)
	customArgs.AddFlags(pflag.CommandLine)
	flag.Set("logtostderr", "true")
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()

	if err := generatorargs.Validate(genericArgs); err != nil {
		klog.Fatalf("Error: %v", err)
	}

	// Run it.
	if err := genericArgs.Execute(
		statusgenerators.NameSystems(),
		statusgenerators.DefaultNameSystem(),
		statusgenerators.Packages,
	); err != nil {
		klog.Fatalf("Error: %v", err)
	}
	klog.V(2).Info("Completed successfully.")
}
