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
)

// Args is used by the gengo framework to pass args specific to this generator.
type Args struct {
	OutputDir                 string // must be a directory path
	OutputPkg                 string // must be a Go import-path
	GoHeaderFile              string
	VersionedClientSetPackage string // must be a Go import-path
	InternalClientSetPackage  string // must be a Go import-path
	ListersPackage            string // must be a Go import-path
	SingleDirectory           bool

	// PluralExceptions define a list of pluralizer exceptions in Type:PluralType format.
	// The default list is "Endpoints:Endpoints"
	PluralExceptions []string
}

// New returns default arguments for the generator.
func New() *Args {
	return &Args{
		SingleDirectory:  false,
		PluralExceptions: []string{"Endpoints:Endpoints"},
	}
}

// AddFlags add the generator flags to the flag set.
func (args *Args) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&args.OutputDir, "output-dir", "",
		"the base directory under which to generate results")
	fs.StringVar(&args.OutputPkg, "output-pkg", args.OutputPkg,
		"the Go import-path of the generated results")
	fs.StringVar(&args.GoHeaderFile, "go-header-file", "",
		"the path to a file containing boilerplate header text; the string \"YEAR\" will be replaced with the current 4-digit year")
	fs.StringVar(&args.InternalClientSetPackage, "internal-clientset-package", args.InternalClientSetPackage,
		"the Go import-path of the internal clientset to use")
	fs.StringVar(&args.VersionedClientSetPackage, "versioned-clientset-package", args.VersionedClientSetPackage,
		"the Go import-path of the versioned clientset to use")
	fs.StringVar(&args.ListersPackage, "listers-package", args.ListersPackage,
		"the Go import-path of the listers to use")
	fs.BoolVar(&args.SingleDirectory, "single-directory", args.SingleDirectory,
		"if true, omit the intermediate \"internalversion\" and \"externalversions\" subdirectories")
	fs.StringSliceVar(&args.PluralExceptions, "plural-exceptions", args.PluralExceptions,
		"list of comma separated plural exception definitions in Type:PluralizedType format")
}

// Validate checks the given arguments.
func (args *Args) Validate() error {
	if len(args.OutputDir) == 0 {
		return fmt.Errorf("--output-dir must be specified")
	}
	if len(args.OutputPkg) == 0 {
		return fmt.Errorf("--output-pkg must be specified")
	}
	if len(args.VersionedClientSetPackage) == 0 {
		return fmt.Errorf("--versioned-clientset-package must be specified")
	}
	if len(args.ListersPackage) == 0 {
		return fmt.Errorf("--listers-package must be specified")
	}
	return nil
}
