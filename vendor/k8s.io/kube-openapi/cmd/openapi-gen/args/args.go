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
	"k8s.io/gengo/args"
)

// CustomArgs is used by the gengo framework to pass args specific to this generator.
type CustomArgs struct {
	// ReportFilename is added to CustomArgs for specifying name of report file used
	// by API linter. If specified, API rule violations will be printed to report file.
	// Otherwise default value "-" will be used which indicates stdout.
	ReportFilename string
}

// NewDefaults returns default arguments for the generator. Returning the arguments instead
// of using default flag parsing allows registering custom arguments afterwards
func NewDefaults() (*args.GeneratorArgs, *CustomArgs) {
	// Default() sets a couple of flag default values for example the boilerplate.
	// WithoutDefaultFlagParsing() disables implicit addition of command line flags and parsing,
	// which allows registering custom arguments afterwards
	genericArgs := args.Default().WithoutDefaultFlagParsing()
	customArgs := &CustomArgs{}
	genericArgs.CustomArgs = customArgs

	// Default value for report filename is "-", which stands for stdout
	customArgs.ReportFilename = "-"
	// Default value for output file base name
	genericArgs.OutputFileBaseName = "openapi_generated"

	return genericArgs, customArgs
}

// AddFlags add the generator flags to the flag set.
func (c *CustomArgs) AddFlags(fs *pflag.FlagSet) {
	fs.StringVarP(&c.ReportFilename, "report-filename", "r", c.ReportFilename, "Name of report file used by API linter to print API violations. Default \"-\" stands for standard output. NOTE that if valid filename other than \"-\" is specified, API linter won't return error on detected API violations. This allows further check of existing API violations without stopping the OpenAPI generation toolchain.")
}

// Validate checks the given arguments.
func Validate(genericArgs *args.GeneratorArgs) error {
	c, ok := genericArgs.CustomArgs.(*CustomArgs)
	if !ok {
		return fmt.Errorf("input arguments don't contain valid custom arguments")
	}
	if len(c.ReportFilename) == 0 {
		return fmt.Errorf("report filename cannot be empty. specify a valid filename or use \"-\" for stdout")
	}
	if len(genericArgs.OutputFileBaseName) == 0 {
		return fmt.Errorf("output file base name cannot be empty")
	}
	if len(genericArgs.OutputPackagePath) == 0 {
		return fmt.Errorf("output package cannot be empty")
	}
	return nil
}
