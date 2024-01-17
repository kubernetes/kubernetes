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

package args

import (
	"fmt"

	"github.com/spf13/pflag"
	"k8s.io/gengo/v2/args"
)

// CustomArgs is used by the gengo framework to pass args specific to this generator.
type CustomArgs struct {
	OutputDir    string // must be a directory path
	OutputPkg    string // must be a Go import-path
	GoHeaderFile string

	// PluralExceptions specify list of exceptions used when pluralizing certain types.
	// For example 'Endpoints:Endpoints', otherwise the pluralizer will generate 'Endpointes'.
	PluralExceptions []string
}

// NewDefaults returns default arguments for the generator.
func NewDefaults() (*args.GeneratorArgs, *CustomArgs) {
	genericArgs := args.Default()
	customArgs := &CustomArgs{
		PluralExceptions: []string{"Endpoints:Endpoints"},
	}
	genericArgs.CustomArgs = customArgs

	return genericArgs, customArgs
}

// AddFlags add the generator flags to the flag set.
func (ca *CustomArgs) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&ca.OutputDir, "output-dir", "",
		"the base directory under which to generate results")
	fs.StringVar(&ca.OutputPkg, "output-pkg", "",
		"the base Go import-path under which to generate results")
	fs.StringSliceVar(&ca.PluralExceptions, "plural-exceptions", ca.PluralExceptions,
		"list of comma separated plural exception definitions in Type:PluralizedType format")
	fs.StringVar(&ca.GoHeaderFile, "go-header-file", "",
		"the path to a file containing boilerplate header text; the string \"YEAR\" will be replaced with the current 4-digit year")
}

// Validate checks the given arguments.
func Validate(genericArgs *args.GeneratorArgs) error {
	custom := genericArgs.CustomArgs.(*CustomArgs)

	if len(custom.OutputDir) == 0 {
		return fmt.Errorf("--output-dir must be specified")
	}
	if len(custom.OutputPkg) == 0 {
		return fmt.Errorf("--output-pkg must be specified")
	}

	return nil
}
