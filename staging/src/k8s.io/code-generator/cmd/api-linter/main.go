/*
Copyright 2017 The Kubernetes Authors.

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

// API linter follows the same rules as openapi-gen, and validates API conventions on every type or package
// that requires API definition generation. The following rules are the same from openapi-gen:
//
// - To generate definition for a specific type or package add "+k8s:openapi-gen=true" tag to the type/package comment lines.
// - To exclude a type or a member from a tagged package/type, add "+k8s:openapi-gen=false" tag to the comment lines.
//
// This directory contains the gengo framework for API linter `api-linter.go` and
// other implementations of API convention validators. Each validator should
// implement the interface `linters.APIValidator`, and be passed to
//
//     func newAPILinter(sanitizedName, filename string, apiValidators []APIValidator) generator.Generator
//
// as part of argument `apiValidators`.

package main

import (
	goflag "flag"
	"fmt"
	"os"

	"k8s.io/code-generator/cmd/api-linter/linters"
	"k8s.io/gengo/args"
	"k8s.io/gengo/generator"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

const helpText = `# API linter

The API linter follows the same rules as openapi-gen, and validates API conventions on every type or package
that requires API definition generation. The following rules are the same from openapi-gen:

- To generate definition for a specific type or package add "+k8s:openapi-gen=true" tag to the type/package comment lines.
- To exclude a type or a member from a tagged package/type, add "+k8s:openapi-gen=false" tag to the comment lines.
`

func main() {
	arguments := args.Default()
	arguments.OutputFileBaseName = "api_linter"

	customArgs := &linters.CustomArgs{
		WhitelistFilename: "",
	}
	pflag.CommandLine.StringVar(&customArgs.WhitelistFilename, "whitelist-file", customArgs.WhitelistFilename, "Specify a csv file that contains the whitelist.")
	arguments.CustomArgs = customArgs
	pflag.Usage = func() {
		fmt.Fprintf(os.Stderr, "%s\n", helpText)
		fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
		pflag.PrintDefaults()
	}

	arguments.AddFlags(pflag.CommandLine)
	pflag.CommandLine.AddGoFlagSet(goflag.CommandLine)
	pflag.Parse()

	b, err := arguments.NewBuilder()
	if err != nil {
		glog.Fatalf("Error: Failed making a parser: %v", err)
	}

	c, err := generator.NewContext(b, linters.NameSystems(), linters.DefaultNameSystem())
	if err != nil {
		glog.Fatalf("Error: Failed making a context: %v", err)
	}

	c.FileTypes["apiLinterReport"] = linters.NewReportFile()
	packages := linters.Packages(c, arguments)
	if err := c.ExecutePackages(arguments.OutputBase, packages); err != nil {
		glog.Fatalf("Error: Failed executing generator: %v", err)
	}
	glog.V(1).Info("API linter finished successfully")
}
