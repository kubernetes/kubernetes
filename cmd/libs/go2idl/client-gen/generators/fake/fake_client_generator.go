/*
Copyright 2015 The Kubernetes Authors.

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

package fake

import (
	"path/filepath"
	"strings"

	"github.com/golang/glog"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/types"
	clientgenargs "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/args"
	clientgentypes "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/types"
)

func PackageForGroup(gv clientgentypes.GroupVersion, typeList []*types.Type, packageBasePath string, srcTreePath string, inputPath string, boilerplate []byte, generatedBy string) generator.Package {
	outputPackagePath := strings.ToLower(filepath.Join(packageBasePath, gv.Group.NonEmpty(), gv.Version.NonEmpty(), "fake"))
	// TODO: should make this a function, called by here and in client-generator.go
	realClientPath := filepath.Join(packageBasePath, gv.Group.NonEmpty(), gv.Version.NonEmpty())
	return &generator.DefaultPackage{
		PackageName: "fake",
		PackagePath: outputPackagePath,
		HeaderText:  boilerplate,
		PackageDocumentation: []byte(
			generatedBy +
				`// Package fake has the automatically generated clients.
`),
		// GeneratorFunc returns a list of generators. Each generator makes a
		// single file.
		GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = []generator.Generator{
				// Always generate a "doc.go" file.
				generator.DefaultGen{OptionalName: "doc"},
			}
			// Since we want a file per type that we generate a client for, we
			// have to provide a function for this.
			for _, t := range typeList {
				generators = append(generators, &genFakeForType{
					DefaultGen: generator.DefaultGen{
						OptionalName: "fake_" + strings.ToLower(c.Namers["private"].Name(t)),
					},
					outputPackage: outputPackagePath,
					group:         gv.Group.NonEmpty(),
					version:       gv.Version.String(),
					inputPackage:  inputPath,
					typeToMatch:   t,
					imports:       generator.NewImportTracker(),
				})
			}

			generators = append(generators, &genFakeForGroup{
				DefaultGen: generator.DefaultGen{
					OptionalName: "fake_" + gv.Group.NonEmpty() + "_client",
				},
				outputPackage:  outputPackagePath,
				realClientPath: realClientPath,
				group:          gv.Group.NonEmpty(),
				version:        gv.Version.String(),
				types:          typeList,
				imports:        generator.NewImportTracker(),
			})
			return generators
		},
		FilterFunc: func(c *generator.Context, t *types.Type) bool {
			return extractBoolTagOrDie("genclient", t.SecondClosestCommentLines) == true
		},
	}
}

func extractBoolTagOrDie(key string, lines []string) bool {
	val, err := types.ExtractSingleBoolCommentTag("+", key, false, lines)
	if err != nil {
		glog.Fatalf(err.Error())
	}
	return val
}

func PackageForClientset(customArgs clientgenargs.Args, typedClientBasePath string, boilerplate []byte, generatedBy string) generator.Package {
	return &generator.DefaultPackage{
		// TODO: we'll generate fake clientset for different release in the future.
		// Package name and path are hard coded for now.
		PackageName: "fake",
		PackagePath: filepath.Join(customArgs.ClientsetOutputPath, customArgs.ClientsetName, "fake"),
		HeaderText:  boilerplate,
		PackageDocumentation: []byte(
			generatedBy +
				`// This package has the automatically generated fake clientset.
`),
		// GeneratorFunc returns a list of generators. Each generator generates a
		// single file.
		GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = []generator.Generator{
				// Always generate a "doc.go" file.
				generator.DefaultGen{OptionalName: "doc"},

				&genClientset{
					DefaultGen: generator.DefaultGen{
						OptionalName: "clientset_generated",
					},
					groups:          customArgs.Groups,
					typedClientPath: typedClientBasePath,
					outputPackage:   "fake",
					imports:         generator.NewImportTracker(),
					clientsetPath:   filepath.Join(customArgs.ClientsetOutputPath, customArgs.ClientsetName),
				},
			}
			return generators
		},
	}
}
