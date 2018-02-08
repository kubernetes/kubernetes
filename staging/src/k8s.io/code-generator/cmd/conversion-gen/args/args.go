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
	"k8s.io/gengo/args"
)

// DefaultBasePeerDirs are the peer-dirs nearly everybody will use, i.e. those coming from
// apimachinery.
var DefaultBasePeerDirs = []string{
	"k8s.io/apimachinery/pkg/apis/meta/v1",
	"k8s.io/apimachinery/pkg/conversion",
	"k8s.io/apimachinery/pkg/runtime",
}

// CustomArgs is used by the gengo framework to pass args specific to this generator.
type CustomArgs struct {
	// Base peer dirs which nearly everybody will use, i.e. outside of Kubernetes core. Peer dirs
	// are declared to make the generator pick up manually written conversion funcs from external
	// packages.
	BasePeerDirs []string

	// Custom peer dirs which are application specific. Peer dirs are declared to make the
	// generator pick up manually written conversion funcs from external packages.
	ExtraPeerDirs []string

	// Skipunsafe indicates whether to generate unsafe conversions to improve the efficiency
	// of these operations. The unsafe operation is a direct pointer assignment via unsafe
	// (within the allowed uses of unsafe) and is equivalent to a proposed Golang change to
	// allow structs that are identical to be assigned to each other.
	SkipUnsafe bool
}

// NewDefaults returns default arguments for the generator.
func NewDefaults() (*args.GeneratorArgs, *CustomArgs) {
	genericArgs := args.Default().WithoutDefaultFlagParsing()
	customArgs := &CustomArgs{
		BasePeerDirs: DefaultBasePeerDirs,
		SkipUnsafe:   false,
	}
	genericArgs.CustomArgs = customArgs
	genericArgs.OutputFileBaseName = "conversion_generated"
	return genericArgs, customArgs
}

// AddFlags add the generator flags to the flag set.
func (ca *CustomArgs) AddFlags(fs *pflag.FlagSet) {
	pflag.CommandLine.StringSliceVar(&ca.BasePeerDirs, "base-peer-dirs", ca.BasePeerDirs,
		"Comma-separated list of apimachinery import paths which are considered, after tag-specified peers, for conversions. Only change these if you have very good reasons.")
	pflag.CommandLine.StringSliceVar(&ca.ExtraPeerDirs, "extra-peer-dirs", ca.ExtraPeerDirs,
		"Application specific comma-separated list of import paths which are considered, after tag-specified peers and base-peer-dirs, for conversions.")
	pflag.CommandLine.BoolVar(&ca.SkipUnsafe, "skip-unsafe", ca.SkipUnsafe,
		"If true, will not generate code using unsafe pointer conversions; resulting code may be slower.")
}

// Validate checks the given arguments.
func Validate(genericArgs *args.GeneratorArgs) error {
	_ = genericArgs.CustomArgs.(*CustomArgs)

	if len(genericArgs.OutputFileBaseName) == 0 {
		return fmt.Errorf("output file base name cannot be empty")
	}

	return nil
}
