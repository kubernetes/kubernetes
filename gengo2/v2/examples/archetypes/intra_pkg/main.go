/*
Copyright 2023 The Kubernetes Authors.

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

// Package main is an example of an archetypal gengo/v2 tool which generates
// code into the same package(s) it used as input(s).
package main

import (
	"fmt"

	"github.com/spf13/pflag"
	"k8s.io/gengo/v2/args"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/klog/v2"
)

func main() {
	klog.InitFlags(nil)
	gengo := getArgs()

	// FIXME: these functions could be named better
	if err := gengo.Execute(getPackageNamers(), getDefaultNamer(), getPackages); err != nil {
		klog.Fatalf("ERROR: %v", err)
	}
	klog.V(2).Info("completed successfully")
}

// getArgs returns the arguments for this tool.
func getArgs() *args.GeneratorArgs {
	//FIXME: rename to something more friendly, like gengo.Args()
	stdArgs := args.Default()
	stdArgs.CustomArgs = &customArgs{}
	stdArgs.Validate = validateArgs
	// Set any other non-standard defaults here.
	return stdArgs
}

// customArgs captures any non-standard arguments this tool needs.
type customArgs struct {
	StringArg string
}

// AddFlagsTo adds the generator flags to the flag set.
func (ca customArgs) AddFlagsTo(fs *pflag.FlagSet) {
	pflag.CommandLine.StringVar(&ca.StringArg, "string-arg", "default",
		"An example string argument")
}

// validateArgs checks the given arguments.
func validateArgs(stdArgs *args.GeneratorArgs) error {
	_ = stdArgs.CustomArgs.(*customArgs)

	// FIXME: should this be standard?
	if len(stdArgs.OutputFileBaseName) == 0 {
		return fmt.Errorf("output file base name cannot be empty")
	}

	return nil
}

// getPackageNamers returns the name systems used by the generators in this package.
func getPackageNamers() namer.NameSystems {
	return nil
}

// getDefaultNamer returns the default name system for ordering the types to be
// processed by the generators in this package.
func getDefaultNamer() string {
	return ""
}

// getPackages returns a set of packages to be processed by this tool.
func getPackages(context *generator.Context, arguments *args.GeneratorArgs) generator.Packages {
	return nil
}
