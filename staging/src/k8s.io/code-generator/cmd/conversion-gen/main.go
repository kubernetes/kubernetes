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

// conversion-gen is a tool for auto-generating functions that convert
// between internal and external types.  A general conversion code
// generation task involves three sets of packages: (1) a set of
// packages containing internal types, (2) a single package containing
// the external types, and (3) a single destination package (i.e.,
// where the generated conversion functions go, and where the
// developer-authored conversion functions are).  The packages
// containing the internal types play the role known as "peer
// packages" in the general code-generation framework of Kubernetes.
//
// For each conversion task, `conversion-gen` will generate functions
// that efficiently convert between same-name types in the two
// (internal, external) packages.  The generated functions include
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
// selected types.  For a top-level object type (i.e., the type of an
// object that will be input to an apiserver), for such an override to
// be used by the apiserver the developer-maintained conversion
// functions must also be registered by invoking the
// `AddConversionFunc`/`AddGeneratedConversionFunc` method of the
// relevant `Scheme` object from k8s.io/apimachinery/pkg/runtime.
//
// `conversion-gen` will scan its `--input-dirs`, looking at the
// package defined in each of those directories for comment tags that
// define a conversion code generation task.  A package requests
// conversion code generation by including one or more comment in the
// package's `doc.go` file (currently anywhere in that file is
// acceptable, but the recommended location is above the `package`
// statement), of the form:
//   // +k8s:conversion-gen=<import-path-of-internal-package>
// This introduces a conversion task, for which the destination
// package is the one containing the file with the tag and the tag
// identifies a package containing internal types.  If there is also a
// tag of the form
//   // +k8s:conversion-gen-external-types=<import-path-of-external-package>
// then it identifies the package containing the external types;
// otherwise they are in the destination package.
//
// For each conversion code generation task, the full set of internal
// packages (AKA peer packages) consists of the ones specified in the
// `k8s:conversion-gen` tags PLUS any specified in the
// `--base-peer-dirs` and `--extra-peer-dirs` flags on the command
// line.
//
// When generating for a package, individual types or fields of structs may opt
// out of Conversion generation by specifying a comment on the of the form:
//   // +k8s:conversion-gen=false
package main

import (
	"flag"
	"path/filepath"

	"github.com/spf13/pflag"
	"k8s.io/gengo/args"
	"k8s.io/klog"

	generatorargs "k8s.io/code-generator/cmd/conversion-gen/args"
	"k8s.io/code-generator/cmd/conversion-gen/generators"
	"k8s.io/code-generator/pkg/util"
)

func main() {
	klog.InitFlags(nil)
	genericArgs, customArgs := generatorargs.NewDefaults()

	// Override defaults.
	// TODO: move this out of conversion-gen
	genericArgs.GoHeaderFilePath = filepath.Join(args.DefaultSourceTree(), util.BoilerplatePath())

	genericArgs.AddFlags(pflag.CommandLine)
	customArgs.AddFlags(pflag.CommandLine)
	flag.Set("logtostderr", "true")
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()

	// k8s.io/apimachinery/pkg/runtime contains a number of manual conversions,
	// that we need to generate conversions.
	// Packages being dependencies of explicitly requested packages are only
	// partially scanned - only types explicitly used are being traversed.
	// Not used functions or types are omitted.
	// Adding this explicitly to InputDirs ensures that the package is fully
	// scanned and all functions are parsed and processed.
	genericArgs.InputDirs = append(genericArgs.InputDirs, "k8s.io/apimachinery/pkg/runtime")

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
