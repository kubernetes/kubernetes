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

// DefaultBasePeerDirs are the peer-dirs nearly everybody will use, i.e. those coming from
// apimachinery.
var DefaultBasePeerDirs = []string{
	"k8s.io/apimachinery/pkg/apis/meta/v1",
	"k8s.io/apimachinery/pkg/conversion",
	"k8s.io/apimachinery/pkg/runtime",
}

type Args struct {
	// The filename of the generated results.
	OutputFile string

	// Base peer dirs which nearly everybody will use, i.e. outside of Kubernetes core. Peer dirs
	// are declared to make the generator pick up manually written conversion funcs from external
	// packages.
	BasePeerDirs []string

	// Custom peer dirs which are application specific. Peer dirs are declared to make the
	// generator pick up manually written conversion funcs from external packages.
	ExtraPeerDirs []string

	// Additional dirs to parse and load, but not consider for peers.  This is
	// useful when packages depend on other packages and want to call
	// conversions across them.
	ExtraDirs []string

	// SkipUnsafe indicates whether to generate unsafe conversions to improve the efficiency
	// of these operations. The unsafe operation is a direct pointer assignment via unsafe
	// (within the allowed uses of unsafe) and is equivalent to a proposed Golang change to
	// allow structs that are identical to be assigned to each other.
	SkipUnsafe bool

	// GoHeaderFile is the path to a boilerplate header file for generated
	// code.
	GoHeaderFile string
}

// New returns default arguments for the generator.
func New() *Args {
	return &Args{
		BasePeerDirs: DefaultBasePeerDirs,
		SkipUnsafe:   false,
	}
}

// AddFlags add the generator flags to the flag set.
func (args *Args) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&args.OutputFile, "output-file", "generated.conversion.go",
		"the name of the file to be generated")
	fs.StringSliceVar(&args.BasePeerDirs, "base-peer-dirs", args.BasePeerDirs,
		"Comma-separated list of apimachinery import paths which are considered, after tag-specified peers, for conversions. Only change these if you have very good reasons.")
	fs.StringSliceVar(&args.ExtraPeerDirs, "extra-peer-dirs", args.ExtraPeerDirs,
		"Application specific comma-separated list of import paths which are considered, after tag-specified peers and base-peer-dirs, for conversions.")
	fs.StringSliceVar(&args.ExtraDirs, "extra-dirs", args.ExtraDirs,
		"Application specific comma-separated list of import paths which are loaded and considered for callable conversions, but are not considered peers for conversion.")
	fs.BoolVar(&args.SkipUnsafe, "skip-unsafe", args.SkipUnsafe,
		"If true, will not generate code using unsafe pointer conversions; resulting code may be slower.")
	fs.StringVar(&args.GoHeaderFile, "go-header-file", "",
		"the path to a file containing boilerplate header text; the string \"YEAR\" will be replaced with the current 4-digit year")
}

// Validate checks the given arguments.
func (args *Args) Validate() error {
	if len(args.OutputFile) == 0 {
		return fmt.Errorf("--output-file must be specified")
	}
	return nil
}
