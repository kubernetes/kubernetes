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

	"k8s.io/gengo/generator"
	"k8s.io/gengo/types"

	clientgenargs "k8s.io/code-generator/cmd/client-gen/args"
	scheme "k8s.io/code-generator/cmd/client-gen/generators/scheme"
	"k8s.io/code-generator/cmd/client-gen/generators/util"
	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"
)

func PackageForGroup(gv clientgentypes.GroupVersion, typeList []*types.Type, clientsetPackage string, groupPackageName string, groupGoName string, inputPackage string, boilerplate []byte) generator.Package {
	outputPackage := filepath.Join(clientsetPackage, "typed", strings.ToLower(groupPackageName), strings.ToLower(gv.Version.NonEmpty()), "fake")
	// TODO: should make this a function, called by here and in client-generator.go
	realClientPackage := filepath.Join(clientsetPackage, "typed", strings.ToLower(groupPackageName), strings.ToLower(gv.Version.NonEmpty()))
	return &generator.DefaultPackage{
		PackageName: "fake",
		PackagePath: outputPackage,
		HeaderText:  boilerplate,
		PackageDocumentation: []byte(
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
					outputPackage: outputPackage,
					inputPackage:  inputPackage,
					group:         gv.Group.NonEmpty(),
					version:       gv.Version.String(),
					groupGoName:   groupGoName,
					typeToMatch:   t,
					imports:       generator.NewImportTracker(),
				})
			}

			generators = append(generators, &genFakeForGroup{
				DefaultGen: generator.DefaultGen{
					OptionalName: "fake_" + groupPackageName + "_client",
				},
				outputPackage:     outputPackage,
				realClientPackage: realClientPackage,
				group:             gv.Group.NonEmpty(),
				version:           gv.Version.String(),
				groupGoName:       groupGoName,
				types:             typeList,
				imports:           generator.NewImportTracker(),
			})
			return generators
		},
		FilterFunc: func(c *generator.Context, t *types.Type) bool {
			return util.MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...)).GenerateClient
		},
	}
}

func PackageForClientset(customArgs *clientgenargs.CustomArgs, clientsetPackage string, groupGoNames map[clientgentypes.GroupVersion]string, boilerplate []byte) generator.Package {
	return &generator.DefaultPackage{
		// TODO: we'll generate fake clientset for different release in the future.
		// Package name and path are hard coded for now.
		PackageName: "fake",
		PackagePath: filepath.Join(clientsetPackage, "fake"),
		HeaderText:  boilerplate,
		PackageDocumentation: []byte(
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
					groups:               customArgs.Groups,
					groupGoNames:         groupGoNames,
					fakeClientsetPackage: clientsetPackage,
					outputPackage:        "fake",
					imports:              generator.NewImportTracker(),
					realClientsetPackage: clientsetPackage,
				},
				&scheme.GenScheme{
					DefaultGen: generator.DefaultGen{
						OptionalName: "register",
					},
					InputPackages: customArgs.GroupVersionPackages(),
					OutputPackage: clientsetPackage,
					Groups:        customArgs.Groups,
					GroupGoNames:  groupGoNames,
					ImportTracker: generator.NewImportTracker(),
					PrivateScheme: true,
				},
			}
			return generators
		},
	}
}
