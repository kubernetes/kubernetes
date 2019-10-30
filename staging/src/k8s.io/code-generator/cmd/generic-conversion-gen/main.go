/*
Copyright 2019 The Kubernetes Authors.

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

// generic-conversion-gen is a tool for auto-generating functions that
// convert between internal and external types.  A general conversion code
// generation task involves three sets of packages: (1) a set of
// packages containing internal types, (2) a single package containing
// the external types, and (3) a single destination package (i.e.,
// where the generated conversion functions go, and where the
// developer-authored conversion functions are).  The packages
// containing the internal types play the role known as "peer
// packages" in the general code-generation framework of Kubernetes.
//
// For each conversion task, `generic-conversion-gen` will generate
// functions that efficiently convert between same-name types in the
// two (internal, external) packages.  The generated functions include
// ones named
//     autoConvert_<pkg1>_<type>_To_<pkg2>_<type>
// for each such pair of types --- both with (pkg1,pkg2) =
// (internal,external) and (pkg1,pkg2) = (external,internal).
// Additionally: if the destination package does not contain one in a
// non-generated file then a function named
//     Convert_<pkg1>_<type>_To_<pkg2>_<type>
// is also generated and it simply calls the `autoConvert...`
// function.  The generated conversion functions use standard value
// assignment wherever possible.  For compound types, the generated
// conversion functions call the `Convert...` functions for the
// subsidiary types.  Thus developers can override the behavior for
// selected types.
//
// When generating for a package, individual types or fields of structs may opt
// out of Conversion generation by specifying a comment on the of the form:
//   // +<tag-name>=false
package main

import (
	"flag"

	"github.com/spf13/pflag"
	"k8s.io/klog"

	generatorargs "k8s.io/code-generator/cmd/generic-conversion-gen/args"
	"k8s.io/code-generator/cmd/generic-conversion-gen/generators"
)

func main() {
	klog.InitFlags(nil)
	genericArgs, customArgs := generatorargs.NewDefaults()

	genericArgs.AddFlags(pflag.CommandLine)
	customArgs.AddFlags(pflag.CommandLine)
	flag.Set("logtostderr", "true")
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()

	if err := generatorargs.Validate(genericArgs); err != nil {
		klog.Fatalf("Error: %v", err)
	}

	if err := genericArgs.Execute(
		generators.NameSystems(),
		generators.DefaultNameSystem(),
		generators.Packages,
	); err != nil {
		klog.Fatalf("Error: %v", err)
	}
	klog.V(2).Info("Completed successfully.")
}
