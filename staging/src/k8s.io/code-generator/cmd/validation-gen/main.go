/*
Copyright 2024 The Kubernetes Authors.

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

// validation-gen is a tool for auto-generating Validation functions.
package main

import (
	"bytes"
	"cmp"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"slices"

	"github.com/spf13/pflag"

	"k8s.io/code-generator/cmd/validation-gen/validators"
	"k8s.io/code-generator/pkg/apidefinitions"
	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"
)

func main() {
	klog.InitFlags(nil)
	args := &Args{}

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
		printDocs()
		os.Exit(0)
	}

	myTargets := func(context *generator.Context) []generator.Target {
		return GetTargets(context, args)
	}

	// Run it.
	if err := gengo.Execute(
		NameSystems(),
		DefaultNameSystem(),
		myTargets,
		gengo.StdBuildTag,
		pflag.Args(),
	); err != nil {
		klog.Fatalf("Error: %v", err)
	}
	klog.V(2).Info("Completed successfully.")
}

type Args struct {
	OutputFile   string
	ReadOnlyPkgs []string // Always consider these as last-ditch possibilities for validations.
	GoHeaderFile string
	PrintDocs    bool
	// TestOutputRoot, when non-empty, enables coverage test fixture
	// generation. For each Kind with declared rules, emits a test directory
	// at <TestOutputRoot>/<short-group>/<lowercase(kind)>/ containing one
	// <TestOutputFilePrefix><version>_test.go per version plus a shared
	// <TestOutputFilePrefix>main_test.go.
	TestOutputRoot string

	// TestOutputFilePrefix is prepended to every emitted test fixture
	// filename. Empty by default; consumers that mark generated files via a
	// linguist-generated gitattributes pattern (e.g. "zz_generated.") set
	// this to that prefix.
	TestOutputFilePrefix string

	// TestAllowlist, when non-empty, is the path to a YAML file of
	// rule-level filters to exclude from fixture generation. Each entry has
	// fields apiVersion, kind, path, errorType, origin (use "*" to wildcard
	// kind/path/errorType/origin) plus a required reason.
	TestAllowlist string

	apidefinitions.LintArgs
}

// AddFlags add the generator flags to the flag set.
func (args *Args) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&args.OutputFile, "output-file", "generated.validations.go",
		"the name of the file to be generated")
	fs.StringSliceVar(&args.ReadOnlyPkgs, "readonly-pkg", args.ReadOnlyPkgs,
		"the import path of a package whose validation can be used by generated code, but is not being generated for")
	fs.StringVar(&args.GoHeaderFile, "go-header-file", "",
		"the path to a file containing boilerplate header text; the string \"YEAR\" will be replaced with the current 4-digit year")
	fs.BoolVar(&args.PrintDocs, "docs", false,
		"print documentation for supported declarative validations, and then exit")
	fs.StringVar(&args.TestOutputRoot, "test-output-root", "",
		"if non-empty, also emit declarative-validation coverage test fixtures under this path, organized as <root>/<group>/<lowercase(kind)>/<file-prefix>{<version>,main}_test.go")
	fs.StringVar(&args.TestOutputFilePrefix, "test-output-file-prefix", "",
		"prefix prepended to every emitted test fixture filename; useful for marking files via a linguist-generated gitattributes pattern (e.g. \"zz_generated.\")")
	fs.StringVar(&args.TestAllowlist, "test-allowlist", "",
		"path to a YAML config file of rule-level filters to exclude from coverage fixture generation; only meaningful with --test-output-root")
	apidefinitions.AddFlags(&args.LintArgs, fs)
}

// Validate checks the given arguments.
func (args *Args) Validate() error {
	if len(args.OutputFile) == 0 {
		return fmt.Errorf("--output-file must be specified")
	}
	if args.TestAllowlist != "" && args.TestOutputRoot == "" {
		return fmt.Errorf("--test-allowlist is only meaningful with --test-output-root")
	}
	if args.TestOutputFilePrefix != "" && args.TestOutputRoot == "" {
		return fmt.Errorf("--test-output-file-prefix is only meaningful with --test-output-root")
	}

	if err := apidefinitions.ValidateFlags(args.LintRules); err != nil {
		return err
	}
	return nil
}

func printDocs() {
	// We need a fake context to init the validator plugins.
	c := &generator.Context{
		Namers:    namer.NameSystems{},
		Universe:  types.Universe{},
		FileTypes: map[string]generator.FileType{},
	}

	// Initialize all registered validators.
	validator := validators.InitGlobalValidator(c)

	docs := validator.Docs()
	for i := range docs {
		d := &docs[i]
		slices.Sort(d.Scopes)
		if d.Usage == "" {
			// Try to generate a usage string if none was provided.
			usage := d.Tag
			if len(d.Args) > 0 {
				usage += "("
				for i := range d.Args {
					if i > 0 {
						usage += ", "
					}
					usage += d.Args[i].Description
				}
				usage += ")"
			}
			if len(d.Payloads) > 0 {
				usage += "="
				if len(d.Payloads) == 1 {
					usage += d.Payloads[0].Description
				} else {
					usage += "<payload>"
				}
			}
			d.Usage = usage
		}
	}
	slices.SortFunc(docs, func(a, b validators.TagDoc) int {
		return cmp.Compare(a.Tag, b.Tag)
	})

	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "    ")
	if err := encoder.Encode(docs); err != nil {
		klog.Fatalf("failed to marshal docs: %v", err)
	}

	fmt.Println(buf.String())
}
