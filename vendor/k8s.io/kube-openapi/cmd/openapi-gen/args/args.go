/*
Copyright 2018 The Kubernetes Authors.

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

	"github.com/spf13/pflag"
)

type Args struct {
	OutputDir  string // must be a directory path
	OutputPkg  string // must be a Go import-path
	OutputFile string

	GoHeaderFile string

	// ReportFilename is added to Args for specifying name of report file used
	// by API linter. If specified, API rule violations will be printed to report file.
	// Otherwise default value "-" will be used which indicates stdout.
	ReportFilename string
}

// New returns default arguments for the generator. Returning the arguments instead
// of using default flag parsing allows registering custom arguments afterwards
func New() *Args {
	args := &Args{}

	// Default value for report filename is "-", which stands for stdout
	args.ReportFilename = "-"

	return args
}

// AddFlags add the generator flags to the flag set.
func (args *Args) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&args.OutputDir, "output-dir", "",
		"the base directory under which to generate results")
	fs.StringVar(&args.OutputPkg, "output-pkg", "",
		"the base Go import-path under which to generate results")
	fs.StringVar(&args.OutputFile, "output-file", "generated.openapi.go",
		"the name of the file to be generated")
	fs.StringVar(&args.GoHeaderFile, "go-header-file", "",
		"the path to a file containing boilerplate header text; the string \"YEAR\" will be replaced with the current 4-digit year")
	fs.StringVarP(&args.ReportFilename, "report-filename", "r", args.ReportFilename,
		"Name of report file used by API linter to print API violations. Default \"-\" stands for standard output. NOTE that if valid filename other than \"-\" is specified, API linter won't return error on detected API violations. This allows further check of existing API violations without stopping the OpenAPI generation toolchain.")
}

// Validate checks the given arguments.
func (args *Args) Validate() error {
	if len(args.OutputDir) == 0 {
		return fmt.Errorf("--output-dir must be specified")
	}
	if len(args.OutputPkg) == 0 {
		return fmt.Errorf("--output-pkg must be specified")
	}
	if len(args.OutputFile) == 0 {
		return fmt.Errorf("--output-file must be specified")
	}
	if len(args.ReportFilename) == 0 {
		return fmt.Errorf("--report-filename must be specified (use \"-\" for stdout)")
	}
	return nil
}
