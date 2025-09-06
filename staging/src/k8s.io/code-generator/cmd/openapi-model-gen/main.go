/*
Copyright 2025 The Kubernetes Authors.

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

// openapi-model-gen is a tool for auto-generating openapi model name functions.
//
// Given a list of input directories, it will create a zz_generated.openapi_model.go file for all beta APIs which indicates the kinds,
// the release it was introduced, the release it will be deprecated, and the release it will be removed.
//
// Generation is governed by comment tags in the source.  Any package may
// request model name generation by including a comment in the file-comments of
// one file, of the form:
//
//	// +k8s:openapi-model-gen=true
//	// +modelPackageName=io.k8s.api.core.v1
//
// Additionally, individual types may request a exact model name generation by including a comment above the type
// declaration:
//
//	// +modelName=io.k8s.api.core.v1.Example
//	type Example struct {
//	}
//
// Note that registration is a whole-package option, and is not available for
// individual types.
package main

import (
	"flag"

	"github.com/spf13/pflag"

	"k8s.io/code-generator/cmd/openapi-model-gen/generators"
	"k8s.io/code-generator/cmd/prerelease-lifecycle-gen/args"
	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/klog/v2"
)

func main() {
	klog.InitFlags(nil)
	args := args.New()

	args.AddFlags(pflag.CommandLine)
	flag.Set("logtostderr", "true")
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()

	if err := args.Validate(); err != nil {
		klog.Fatalf("Error: %v", err)
	}

	myTargets := func(context *generator.Context) []generator.Target {
		return generators.GetTargets(context, args)
	}

	// Run it.
	if err := gengo.Execute(
		generators.NameSystems(),
		generators.DefaultNameSystem(),
		myTargets,
		gengo.StdBuildTag,
		pflag.Args(),
	); err != nil {
		klog.Fatalf("Error: %v", err)
	}
	klog.V(2).Info("Completed successfully.")
}
