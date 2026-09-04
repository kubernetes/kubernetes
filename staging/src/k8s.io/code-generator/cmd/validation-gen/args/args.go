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

package args

import (
	"fmt"
	"regexp"

	"github.com/spf13/pflag"

	"k8s.io/code-generator/pkg/apidefinitions"
)

// DefaultTagPrefix is the tag prefix used by Kubernetes: "+k8s:required",
// "+k8s:validation-gen", and so on.
const DefaultTagPrefix = "k8s:"

// tagPrefixRE matches an empty prefix or one or more ':'-terminated tag name
// segments, as allowed by the gengo codetags grammar.
var tagPrefixRE = regexp.MustCompile(`^([a-zA-Z_][a-zA-Z0-9_.-]*:)*$`)

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

	// TagPrefix qualifies every tag this generator recognizes, both the
	// package-level tags that configure generation (e.g. "+k8s:validation-gen")
	// and the validation tags themselves (e.g. "+k8s:required"). It is empty
	// or one or more ':'-terminated segments. Generators built on
	// validation-gen set this to claim their own tag namespace.
	TagPrefix string

	apidefinitions.LintArgs
}

// New returns default arguments for the generator.
func New() *Args {
	return &Args{
		TagPrefix: DefaultTagPrefix,
	}
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
	fs.StringVar(&args.TagPrefix, "tag-prefix", args.TagPrefix,
		"the prefix of every tag this generator recognizes, e.g. \"k8s:\" for +k8s:validation-gen and +k8s:required; empty or one or more ':'-terminated segments")
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
	if !tagPrefixRE.MatchString(args.TagPrefix) {
		return fmt.Errorf("--tag-prefix %q must be empty or one or more ':'-terminated tag name segments (e.g. \"k8s:\")", args.TagPrefix)
	}

	if err := apidefinitions.ValidateFlags(args.LintRules); err != nil {
		return err
	}
	return nil
}
