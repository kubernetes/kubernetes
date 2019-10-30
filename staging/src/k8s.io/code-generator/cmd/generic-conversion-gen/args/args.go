/*
Copyright 2019 The Kubernetes Authors.

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
	// Packages containing peer types. Can also contain custom conversion functions.
	PeerPkgs []string

	// TagName allows setting the tag name, ie the marker that this generator
	// will look for in comments on types.
	// * "+<tag-name>=false" in a type's comment will instruct conversion-gen to skip that type.
	TagName string

	// FunctionTagName allows setting the function tag name, ie the marker that this generator
	// will look for in comments on manual conversion functions. In a function's comments:
	// * "+<tag-name>=copy-only" : copy-only functions that are directly assignable can be inlined
	// 	 instead of invoked. As an example, conversion functions exist that allow types with private
	//   fields to be correctly copied between types. These functions are equivalent to a memory assignment,
	//	 and are necessary for the reflection path, but should not block memory conversion.
	// * "+<tag-name>=drop" means to drop that conversion altogether.
	FunctionTagName string

	// SkipUnsafe indicates whether to generate unsafe conversions to improve the efficiency
	// of these operations. The unsafe operation is a direct pointer assignment via unsafe
	// (within the allowed uses of unsafe) and is equivalent to a proposed Golang change to
	// allow structs that are identical to be assigned to each other.
	SkipUnsafe bool
}

// NewDefaults returns default arguments for the generator.
func NewDefaults() (*args.GeneratorArgs, *CustomArgs) {
	genericArgs := args.Default().WithoutDefaultFlagParsing()
	customArgs := &CustomArgs{
		SkipUnsafe: false,
	}
	genericArgs.CustomArgs = customArgs
	genericArgs.OutputFileBaseName = "conversion_generated"
	return genericArgs, customArgs
}

// AddFlags add the generator flags to the flag set.
func (ca *CustomArgs) AddFlags(fs *pflag.FlagSet) {
	pflag.CommandLine.StringSliceVar(&ca.PeerPkgs, "peer-pkgs", ca.PeerPkgs,
		"Packages containing peer types. Can also contain custom conversion functions..")
	pflag.CommandLine.StringVar(&ca.TagName, "tag-name", ca.TagName,
		"Tag name to look for in comments on types, e.g. to opt a type out of conversion")
	pflag.CommandLine.StringVar(&ca.FunctionTagName, "function-tag-name", ca.FunctionTagName,
		"Tag name that to look for in comments on conversion functions, e.g. to not use one for generated code.")
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
