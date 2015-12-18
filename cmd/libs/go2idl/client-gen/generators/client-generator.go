/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// Package generators has the generators for the client-gen utility.
package generators

import (
	"path/filepath"

	"k8s.io/kubernetes/cmd/libs/go2idl/args"
	"k8s.io/kubernetes/cmd/libs/go2idl/generator"
	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"

	"github.com/golang/glog"
)

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	pluralExceptions := map[string]string{
		"Endpoints":       "Endpoints",
		"ComponentStatus": "ComponentStatus",
	}
	return namer.NameSystems{
		"public":        namer.NewPublicNamer(0),
		"private":       namer.NewPrivateNamer(0),
		"raw":           namer.NewRawNamer("", nil),
		"publicPlural":  namer.NewPublicPluralNamer(pluralExceptions),
		"privatePlural": namer.NewPrivatePluralNamer(pluralExceptions),
	}
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "public"
}

func packageForGroup(group string, version string, typeList []*types.Type, basePath string, boilerplate []byte) generator.Package {
	outputPackagePath := filepath.Join(basePath, group, version)
	return &generator.DefaultPackage{
		PackageName: version,
		PackagePath: outputPackagePath,
		HeaderText:  boilerplate,
		PackageDocumentation: []byte(
			`// Package unversioned has the automatically generated clients for unversioned resources.
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
				generators = append(generators, &genClientForType{
					DefaultGen: generator.DefaultGen{
						// Use the privatized version of the
						// type name as the file name.
						//
						// TODO: make a namer that converts
						// camelCase to '-' separation for file
						// names?
						OptionalName: c.Namers["private"].Name(t),
					},
					outputPackage: outputPackagePath,
					group:         group,
					typeToMatch:   t,
					imports:       generator.NewImportTracker(),
				})
			}

			generators = append(generators, &genGroup{
				DefaultGen: generator.DefaultGen{
					OptionalName: group + "_client",
				},
				outputPackage: outputPackagePath,
				group:         group,
				types:         typeList,
				imports:       generator.NewImportTracker(),
			})
			return generators
		},
		FilterFunc: func(c *generator.Context, t *types.Type) bool {
			return types.ExtractCommentTags("+", t.SecondClosestCommentLines)["genclient"] == "true"
		},
	}
}

// Packages makes the client package definition.
func Packages(context *generator.Context, arguments *args.GeneratorArgs) generator.Packages {
	boilerplate, err := arguments.LoadGoBoilerplate()
	if err != nil {
		glog.Fatalf("Failed loading boilerplate: %v", err)
	}

	groupToTypes := map[string][]*types.Type{}
	for _, inputDir := range arguments.InputDirs {
		p := context.Universe.Package(inputDir)
		for _, t := range p.Types {
			if types.ExtractCommentTags("+", t.SecondClosestCommentLines)["genclient"] != "true" {
				continue
			}
			group := filepath.Base(t.Name.Package)
			// Special case for the legacy API.
			if group == "api" {
				group = "legacy"
			}
			if _, found := groupToTypes[group]; !found {
				groupToTypes[group] = []*types.Type{}
			}
			groupToTypes[group] = append(groupToTypes[group], t)
		}
	}

	var packageList []generator.Package
	orderer := namer.Orderer{namer.NewPrivateNamer(0)}
	for group, types := range groupToTypes {
		packageList = append(packageList, packageForGroup(group, "unversioned", orderer.OrderTypes(types), arguments.OutputPackagePath, boilerplate))
	}

	return generator.Packages(packageList)
}
