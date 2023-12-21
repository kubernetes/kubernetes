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
	OutputPackage             string // must be a Go import-path
	VersionedClientSetPackage string // must be a Go import-path
	InternalClientSetPackage  string // must be a Go import-path
	ListersPackage            string // must be a Go import-path
	SingleDirectory           bool

	// PluralExceptions define a list of pluralizer exceptions in Type:PluralType format.
	// The default list is "Endpoints:Endpoints"
	PluralExceptions []string
}

// NewDefaults returns default arguments for the generator.
func NewDefaults() (*args.GeneratorArgs, *CustomArgs) {
	genericArgs := args.Default().WithoutDefaultFlagParsing()
	customArgs := &CustomArgs{
		SingleDirectory:  false,
		PluralExceptions: []string{"Endpoints:Endpoints"},
	}
	genericArgs.CustomArgs = customArgs

	return genericArgs, customArgs
}

// AddFlags add the generator flags to the flag set.
func (ca *CustomArgs) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&ca.OutputPackage, "output-package", ca.OutputPackage, "the Go import-path of the generated results")
	fs.StringVar(&ca.InternalClientSetPackage, "internal-clientset-package", ca.InternalClientSetPackage, "the Go import-path of the internal clientset to use")
	fs.StringVar(&ca.VersionedClientSetPackage, "versioned-clientset-package", ca.VersionedClientSetPackage, "the Go import-path of the versioned clientset to use")
	fs.StringVar(&ca.ListersPackage, "listers-package", ca.ListersPackage, "the Go import-path of the listers to use")
	fs.BoolVar(&ca.SingleDirectory, "single-directory", ca.SingleDirectory, "if true, omit the intermediate \"internalversion\" and \"externalversions\" subdirectories")
	fs.StringSliceVar(&ca.PluralExceptions, "plural-exceptions", ca.PluralExceptions, "list of comma separated plural exception definitions in Type:PluralizedType format")
}

// Validate checks the given arguments.
func Validate(genericArgs *args.GeneratorArgs) error {
	if len(genericArgs.OutputBase) == 0 {
		return fmt.Errorf("--output-base must be specified")
	}

	customArgs := genericArgs.CustomArgs.(*CustomArgs)

	if len(customArgs.OutputPackage) == 0 {
		return fmt.Errorf("--output-package must be specified")
	}
	if len(customArgs.VersionedClientSetPackage) == 0 {
		return fmt.Errorf("--versioned-clientset-package must be specified")
	}
	if len(customArgs.ListersPackage) == 0 {
		return fmt.Errorf("--listers-package must be specified")
	}

	return nil
}
