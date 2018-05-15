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

// conversion-gen is a tool for auto-generating Conversion functions.
//
// Given a list of input directories, it will scan for "peer" packages and
// generate functions that efficiently convert between same-name types in each
// package.  For any pair of types that has a
//     `Convert_<pkg1>_<type>_To_<pkg2>_<Type()`
// function (and its reciprocal), it will simply call that.  use standard value
// assignment whenever possible.  The resulting file will be stored in the same
// directory as the processed source package.
//
// Generation is governed by comment tags in the source.  Any package may
// request Conversion generation by including a comment in the file-comments of
// one file, of the form:
//   // +k8s:conversion-gen=<import-path-of-peer-package>
//
// When generating for a package, individual types or fields of structs may opt
// out of Conversion generation by specifying a comment on the of the form:
//   // +k8s:conversion-gen=false
package main

import (
	"flag"
	"path/filepath"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
	"k8s.io/gengo/args"

	generatorargs "k8s.io/code-generator/cmd/conversion-gen/args"
	"k8s.io/code-generator/cmd/conversion-gen/generators"
	"k8s.io/code-generator/pkg/util"
)

func main() {
	genericArgs, customArgs := generatorargs.NewDefaults()

	// Override defaults.
	// TODO: move this out of conversion-gen
	genericArgs.GoHeaderFilePath = filepath.Join(args.DefaultSourceTree(), util.BoilerplatePath())

	genericArgs.AddFlags(pflag.CommandLine)
	customArgs.AddFlags(pflag.CommandLine)
	flag.Set("logtostderr", "true")
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()

	if err := generatorargs.Validate(genericArgs); err != nil {
		glog.Fatalf("Error: %v", err)
	}

	// Run it.
	if err := genericArgs.Execute(
		generators.NameSystems(),
		generators.DefaultNameSystem(),
		generators.Packages,
	); err != nil {
		glog.Fatalf("Error: %v", err)
	}
	glog.V(2).Info("Completed successfully.")
}
