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
	"os"
	"path/filepath"
	"strings"

	"k8s.io/kubernetes/cmd/libs/go2idl/args"
	clientgenargs "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/args"
	"k8s.io/kubernetes/cmd/libs/go2idl/client-gen/generators/fake"
	"k8s.io/kubernetes/cmd/libs/go2idl/client-gen/generators/normalization"
	"k8s.io/kubernetes/cmd/libs/go2idl/generator"
	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"
	"k8s.io/kubernetes/pkg/api/unversioned"

	"github.com/golang/glog"
)

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	pluralExceptions := map[string]string{
		"Endpoints": "Endpoints",
	}
	return namer.NameSystems{
		"public":             namer.NewPublicNamer(0),
		"private":            namer.NewPrivateNamer(0),
		"raw":                namer.NewRawNamer("", nil),
		"publicPlural":       namer.NewPublicPluralNamer(pluralExceptions),
		"privatePlural":      namer.NewPrivatePluralNamer(pluralExceptions),
		"allLowercasePlural": namer.NewAllLowercasePluralNamer(pluralExceptions),
	}
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "public"
}

func packageForGroup(gv unversioned.GroupVersion, typeList []*types.Type, packageBasePath string, srcTreePath string, boilerplate []byte) generator.Package {
	outputPackagePath := filepath.Join(packageBasePath, gv.Group, gv.Version)
	return &generator.DefaultPackage{
		PackageName: gv.Version,
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
						OptionalName: strings.ToLower(c.Namers["private"].Name(t)),
					},
					outputPackage: outputPackagePath,
					group:         gv.Group,
					typeToMatch:   t,
					imports:       generator.NewImportTracker(),
				})
			}

			generators = append(generators, &genGroup{
				DefaultGen: generator.DefaultGen{
					OptionalName: gv.Group + "_client",
				},
				outputPackage: outputPackagePath,
				group:         gv.Group,
				types:         typeList,
				imports:       generator.NewImportTracker(),
			})

			expansionFileName := "generated_expansion"
			// To avoid overriding user's manual modification, only generate the expansion file if it doesn't exist.
			if _, err := os.Stat(filepath.Join(srcTreePath, outputPackagePath, expansionFileName+".go")); os.IsNotExist(err) {
				generators = append(generators, &genExpansion{
					DefaultGen: generator.DefaultGen{
						OptionalName: expansionFileName,
					},
					types: typeList,
				})
			}

			return generators
		},
		FilterFunc: func(c *generator.Context, t *types.Type) bool {
			return types.ExtractCommentTags("+", t.SecondClosestCommentLines)["genclient"] == "true"
		},
	}
}

func packageForClientset(customArgs clientgenargs.Args, typedClientBasePath string, boilerplate []byte) generator.Package {
	return &generator.DefaultPackage{
		PackageName: customArgs.ClientsetName,
		PackagePath: filepath.Join(customArgs.ClientsetOutputPath, customArgs.ClientsetName),
		HeaderText:  boilerplate,
		PackageDocumentation: []byte(
			`// This package has the automatically generated clientset.
`),
		// GeneratorFunc returns a list of generators. Each generator generates a
		// single file.
		GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = []generator.Generator{
				&genClientset{
					DefaultGen: generator.DefaultGen{
						OptionalName: "clientset",
					},
					groupVersions:   customArgs.GroupVersions,
					typedClientPath: typedClientBasePath,
					outputPackage:   customArgs.ClientsetName,
					imports:         generator.NewImportTracker(),
				},
			}
			return generators
		},
	}
}

// Packages makes the client package definition.
func Packages(context *generator.Context, arguments *args.GeneratorArgs) generator.Packages {
	boilerplate, err := arguments.LoadGoBoilerplate()
	if err != nil {
		glog.Fatalf("Failed loading boilerplate: %v", err)
	}

	customArgs, ok := arguments.CustomArgs.(clientgenargs.Args)
	if !ok {
		glog.Fatalf("cannot convert arguments.CustomArgs to clientgenargs.Args")
	}

	gvToTypes := map[unversioned.GroupVersion][]*types.Type{}
	for gv, inputDir := range customArgs.GroupVersionToInputPath {
		p := context.Universe.Package(inputDir)
		for _, t := range p.Types {
			if types.ExtractCommentTags("+", t.SecondClosestCommentLines)["genclient"] != "true" {
				continue
			}
			if _, found := gvToTypes[gv]; !found {
				gvToTypes[gv] = []*types.Type{}
			}
			gvToTypes[gv] = append(gvToTypes[gv], t)
		}
	}

	var packageList []generator.Package

	packageList = append(packageList, packageForClientset(customArgs, arguments.OutputPackagePath, boilerplate))
	if customArgs.FakeClient {
		packageList = append(packageList, fake.PackageForClientset(customArgs, arguments.OutputPackagePath, boilerplate))
	}

	// If --clientset-only=true, we don't regenerate the individual typed clients.
	if customArgs.ClientsetOnly {
		return generator.Packages(packageList)
	}

	orderer := namer.Orderer{namer.NewPrivateNamer(0)}
	for _, gv := range customArgs.GroupVersions {
		types := gvToTypes[gv]
		packageList = append(packageList, packageForGroup(normalization.GroupVersion(gv), orderer.OrderTypes(types), arguments.OutputPackagePath, arguments.OutputBase, boilerplate))
		if customArgs.FakeClient {
			packageList = append(packageList, fake.PackageForGroup(normalization.GroupVersion(gv), orderer.OrderTypes(types), arguments.OutputPackagePath, arguments.OutputBase, boilerplate))
		}
	}

	return generator.Packages(packageList)
}
