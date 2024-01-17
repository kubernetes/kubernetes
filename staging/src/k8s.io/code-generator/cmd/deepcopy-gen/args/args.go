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

package args

import (
	"fmt"

	"github.com/spf13/pflag"
	"k8s.io/gengo/v2/args"
	"k8s.io/gengo/v2/examples/deepcopy-gen/generators"
)

// CustomArgs is used by the gengo framework to pass args specific to this generator.
type CustomArgs generators.CustomArgs

// NewDefaults returns default arguments for the generator.
func NewDefaults() (*args.GeneratorArgs, *CustomArgs) {
	genericArgs := args.Default()
	customArgs := &CustomArgs{}
	genericArgs.CustomArgs = (*generators.CustomArgs)(customArgs) // convert to upstream type to make type-casts work there
	return genericArgs, customArgs
}

// AddFlags add the generator flags to the flag set.
func (ca *CustomArgs) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&ca.OutputFile, "output-file", "generated.deepcopy.go",
		"the name of the file to be generated")
	fs.StringSliceVar(&ca.BoundingDirs, "bounding-dirs", ca.BoundingDirs,
		"Comma-separated list of import paths which bound the types for which deep-copies will be generated.")
	fs.StringVar(&ca.GoHeaderFile, "go-header-file", "",
		"the path to a file containing boilerplate header text; the string \"YEAR\" will be replaced with the current 4-digit year")
}

// Validate checks the given arguments.
func Validate(genericArgs *args.GeneratorArgs) error {
	custom := genericArgs.CustomArgs.(*generators.CustomArgs)

	if len(custom.OutputFile) == 0 {
		return fmt.Errorf("--output-file must be specified")
	}

	return nil
}
