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

// defaulter-gen is a tool for auto-generating Defaulter functions.
//
// Given a list of input directories, it will scan for top level types
// and generate efficient defaulters for an entire object from the sum
// of the SetDefault_* methods contained in the object tree.
//
// Generation is governed by comment tags in the source.  Any package may
// request defaulter generation by including one or more comment tags at
// the package comment level:
//
//   // +k8s:defaulter-gen=<field-name-to-flag>
//
// which will create defaulters for any type that contains the provided
// field name (if the type has defaulters). Any type may request explicit
// defaulting by providing the comment tag:
//
//   // +k8s:defaulter-gen=true|false
//
// An existing defaulter method (`SetDefaults_TYPE`) can provide the
// comment tag:
//
//   // +k8s:defaulter-gen=covers
//
// to indicate that the defaulter does not or should not call any nested
// defaulters.
package main

import (
	"flag"
	"path/filepath"

	"github.com/spf13/pflag"
	"k8s.io/gengo/args"
	"k8s.io/gengo/examples/defaulter-gen/generators"
	"k8s.io/klog/v2"

	generatorargs "k8s.io/code-generator/cmd/defaulter-gen/args"
	"k8s.io/code-generator/pkg/util"
)

func main() {
	klog.InitFlags(nil)
	genericArgs, customArgs := generatorargs.NewDefaults()

	// Override defaults.
	// TODO: move this out of defaulter-gen
	genericArgs.GoHeaderFilePath = filepath.Join(args.DefaultSourceTree(), util.BoilerplatePath())

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
		generators.NameSystems(),
		generators.DefaultNameSystem(),
		generators.Packages,
	); err != nil {
		klog.Fatalf("Error: %v", err)
	}
	klog.V(2).Info("Completed successfully.")
}
