/*
Copyright The Kubernetes Authors.

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

// This is validation-gen built for a hypothetical project whose tags use the
// "xyz:" prefix. It recognizes every standard validation tag under that
// prefix (+xyz:required, +xyz:minimum, ...) plus the project's own tags,
// which are registered by importing the tags package.
package main

import (
	"flag"
	"os"

	"github.com/spf13/pflag"

	"k8s.io/code-generator/cmd/validation-gen/args"
	"k8s.io/code-generator/cmd/validation-gen/generators"
	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/klog/v2"

	// Registers this project's tags with the validation-gen tag registry.
	_ "k8s.io/code-generator/cmd/validation-gen/examples/custom-prefix/tags"
)

func main() {
	klog.InitFlags(nil)
	args := args.New()
	// Claim the project's tag prefix. This is the default for --tag-prefix.
	args.TagPrefix = "xyz:"

	args.AddFlags(pflag.CommandLine)
	if err := flag.Set("logtostderr", "true"); err != nil {
		klog.Fatalf("Error: %v", err)
	}
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()

	if err := args.Validate(); err != nil {
		klog.Fatalf("Error: %v", err)
	}

	if args.PrintDocs {
		if err := generators.PrintDocs(os.Stdout, args.TagPrefix); err != nil {
			klog.Fatalf("Error: %v", err)
		}
		os.Exit(0)
	}

	myTargets := func(context *generator.Context) []generator.Target {
		return generators.GetTargets(context, args)
	}

	if err := gengo.Execute(
		generators.NameSystems(),
		generators.DefaultNameSystem(),
		myTargets,
		gengo.StdBuildTag,
		pflag.Args(),
	); err != nil {
		klog.Fatalf("Error: %v", err)
	}
}
