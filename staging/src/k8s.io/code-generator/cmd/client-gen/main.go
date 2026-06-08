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

// client-gen makes the individual typed clients using gengo.
package main

import (
	"flag"
	"slices"

	"github.com/spf13/pflag"
	"k8s.io/klog/v2"

	"k8s.io/code-generator/cmd/client-gen/args"
	"k8s.io/code-generator/cmd/client-gen/generators"
	"k8s.io/code-generator/pkg/util"
	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
)

func main() {
	klog.InitFlags(nil)
	args := args.New()

	args.AddFlags(pflag.CommandLine, "k8s.io/kubernetes/pkg/apis") // TODO: move this input path out of client-gen
	flag.Set("logtostderr", "true")
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()

	// add group version package as input dirs for gengo
	inputPkgs := []string{}
	for _, pkg := range args.Groups {
		for _, v := range pkg.Versions {
			inputPkgs = append(inputPkgs, v.Package)
		}
	}
	// ensure stable code generation output
	slices.Sort(inputPkgs)

	if err := args.Validate(); err != nil {
		klog.Fatalf("Error: %v", err)
	}

	myTargets := func(context *generator.Context) []generator.Target {
		return generators.GetTargets(context, args)
	}

	if err := gengo.Execute(
		generators.NameSystems(util.PluralExceptionListToMapOrDie(args.PluralExceptions)),
		generators.DefaultNameSystem(),
		myTargets,
		gengo.StdBuildTag,
		inputPkgs,
	); err != nil {
		klog.Fatalf("Error: %v", err)
	}
}
