/*
Copyright 2023 The Kubernetes Authors.

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

package generators

import (
	"github.com/spf13/pflag"
	generatorargs "k8s.io/code-generator/cmd/deepcopy-gen/args"
	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
)

// GenerateDeepCopy runs the deepcopy-gen with given flagset and process args.
func GenerateDeepCopy(fs *pflag.FlagSet, processArgs []string) error {
	args := generatorargs.New()
	args.AddFlags(fs)

	if err := fs.Parse(processArgs); err != nil {
		return err
	}

	if err := args.Validate(); err != nil {
		return err
	}

	myTargets := func(context *generator.Context) []generator.Target {
		return GetTargets(context, args)
	}

	// Run it.
	return gengo.Execute(
		NameSystems(),
		DefaultNameSystem(),
		myTargets,
		gengo.StdBuildTag,
		fs.Args(),
	)
}
