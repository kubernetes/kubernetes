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
	OutputFile   string
	GoHeaderFile string
}

// New returns default arguments for the generator.
func New() *Args {
	return &Args{}
}

// AddFlags add the generator flags to the flag set.
func (args *Args) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&args.OutputFile, "output-file", "generated.register.go",
		"the name of the file to be generated")
	fs.StringVar(&args.GoHeaderFile, "go-header-file", "",
		"the path to a file containing boilerplate header text; the string \"YEAR\" will be replaced with the current 4-digit year")
}

// Validate checks the given arguments.
func (args *Args) Validate() error {
	if len(args.OutputFile) == 0 {
		return fmt.Errorf("output file base name cannot be empty")
	}

	return nil
}
