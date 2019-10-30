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

package generators

import (
	"path/filepath"

	"k8s.io/gengo/args"
	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"

	"k8s.io/klog"

	conversionargs "k8s.io/code-generator/cmd/generic-conversion-gen/args"
)

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	return namer.NameSystems{
		"conversion": ConversionNamer(),
	}
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "conversion"
}

// Packages returns a list of package generators.
func Packages(context *generator.Context, arguments *args.GeneratorArgs) generator.Packages {
	customArgs := &conversionargs.CustomArgs{}
	if args, ok := arguments.CustomArgs.(*conversionargs.CustomArgs); ok {
		customArgs = args
	}

	packages := generator.Packages{}
	manualConversionsTracker := NewManualConversionsTracker()

	processed := map[string]bool{}
	for _, i := range context.Inputs {
		// skip duplicates
		if processed[i] {
			continue
		}
		processed[i] = true

		klog.V(5).Infof("considering pkg %q", i)
		pkg := context.Universe[i]
		if pkg == nil {
			// If the input had no Go files, for example.
			continue
		}

		conversionGenerator, err := NewConversionGenerator(context, arguments.OutputFileBaseName, pkg.Path, pkg.Path,
			customArgs.PeerPkgs, manualConversionsTracker)
		if err != nil {
			klog.Fatalf(err.Error())
		}
		conversionGenerator.WithTagName(customArgs.TagName).
			WithFunctionTagName(customArgs.FunctionTagName).
			WithUnsafeConversions(!customArgs.SkipUnsafe)

		packages = append(packages,
			&generator.DefaultPackage{
				PackageName: filepath.Base(pkg.Path),
				PackagePath: pkg.Path,
				GeneratorList: []generator.Generator{
					conversionGenerator,
				},
				FilterFunc: func(c *generator.Context, t *types.Type) bool {
					return t.Name.Package == pkg.Path
				},
			})

	}

	return packages
}
