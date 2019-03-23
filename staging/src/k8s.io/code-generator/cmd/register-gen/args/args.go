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

	"k8s.io/gengo/args"
)

// NewDefaults returns default arguments for the generator.
func NewDefaults() *args.GeneratorArgs {
	genericArgs := args.Default().WithoutDefaultFlagParsing()
	genericArgs.OutputFileBaseName = "zz_generated.register"
	return genericArgs
}

// Validate checks the given arguments.
func Validate(genericArgs *args.GeneratorArgs) error {
	if len(genericArgs.OutputFileBaseName) == 0 {
		return fmt.Errorf("output file base name cannot be empty")
	}

	return nil
}
