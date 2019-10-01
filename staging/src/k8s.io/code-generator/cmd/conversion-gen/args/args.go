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

	"k8s.io/gengo/args"
	"k8s.io/gengo/examples/conversion-gen/generators"
)

// DefaultBasePeerDirs are the peer-dirs nearly everybody will use, i.e. those coming from
// apimachinery.
var DefaultBasePeerDirs = []string{
	"k8s.io/apimachinery/pkg/apis/meta/v1",
	"k8s.io/apimachinery/pkg/conversion",
	"k8s.io/apimachinery/pkg/runtime",
}

// NewDefaults returns default arguments for the generator.
func NewDefaults() (*args.GeneratorArgs, *generators.CustomArgs) {
	genericArgs := args.Default().WithoutDefaultFlagParsing()
	customArgs := &generators.CustomArgs{
		BasePeerDirs: DefaultBasePeerDirs,
		SkipUnsafe:   false,
	}
	genericArgs.CustomArgs = customArgs
	genericArgs.OutputFileBaseName = "conversion_generated"
	return genericArgs, customArgs
}

// Validate checks the given arguments.
func Validate(genericArgs *args.GeneratorArgs) error {
	_ = genericArgs.CustomArgs.(*generators.CustomArgs)

	if len(genericArgs.OutputFileBaseName) == 0 {
		return fmt.Errorf("output file base name cannot be empty")
	}

	return nil
}
