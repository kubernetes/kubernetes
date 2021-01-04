/*
Copyright 2021 The Kubernetes Authors.

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
	"path"
	"path/filepath"
	"strings"

	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"
	informergenargs "k8s.io/code-generator/cmd/injection-gen/args"
	"k8s.io/gengo/args"
	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
	"k8s.io/klog/v2"
)

// Packages makes the client package definition.
func Packages(context *generator.Context, arguments *args.GeneratorArgs) generator.Packages {
	boilerplate, err := arguments.LoadGoBoilerplate()
	if err != nil {
		klog.Fatal("Failed loading boilerplate: ", err)
	}

	customArgs, ok := arguments.CustomArgs.(*informergenargs.CustomArgs)
	if !ok {
		klog.Fatalf("Wrong CustomArgs type: %T", arguments.CustomArgs)
	}

	versionPackagePath := filepath.Join(arguments.OutputPackagePath)

	var packageList generator.Packages

	groupVersions := make(map[string]clientgentypes.GroupVersions)
	groupGoNames := make(map[string]string)
	for _, inputDir := range arguments.InputDirs {
		p := context.Universe.Package(vendorless(inputDir))

		var gv clientgentypes.GroupVersion
		var targetGroupVersions map[string]clientgentypes.GroupVersions

		parts := strings.Split(p.Path, "/")
		gv.Group = clientgentypes.Group(parts[len(parts)-2])
		gv.Version = clientgentypes.Version(parts[len(parts)-1])
		targetGroupVersions = groupVersions

		groupPackageName := gv.Group.NonEmpty()
		gvPackage := path.Clean(p.Path)

		// If there's a comment of the form "// +groupName=somegroup" or
		// "// +groupName=somegroup.foo.bar.io", use the first field (somegroup) as the name of the
		// group when generating.
		if override := types.ExtractCommentTags("+", p.Comments)["groupName"]; override != nil {
			gv.Group = clientgentypes.Group(override[0])
		}

		// If there's a comment of the form "// +groupGoName=SomeUniqueShortName", use that as
		// the Go group identifier in CamelCase. It defaults
		groupGoNames[groupPackageName] = namer.IC(strings.SplitN(gv.Group.NonEmpty(), ".", 2)[0])
		if override := types.ExtractCommentTags("+", p.Comments)["groupGoName"]; override != nil {
			groupGoNames[groupPackageName] = namer.IC(override[0])
		}

		// Generate the client and fake.
		packageList = append(packageList, versionClientsPackages(versionPackagePath, boilerplate, customArgs)...)

		// Generate the informer factory and fake.
		packageList = append(packageList, versionFactoryPackages(versionPackagePath, boilerplate, customArgs)...)

		var typesWithInformers []*types.Type
		for _, t := range p.Types {
			tags := MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
			if tags.NeedsInformerInjection() {
				typesWithInformers = append(typesWithInformers, t)
			}
		}

		groupVersionsEntry, ok := targetGroupVersions[groupPackageName]
		if !ok {
			groupVersionsEntry = clientgentypes.GroupVersions{
				PackageName: groupPackageName,
				Group:       gv.Group,
			}
		}
		groupVersionsEntry.Versions = append(groupVersionsEntry.Versions, clientgentypes.PackageVersion{Version: gv.Version, Package: gvPackage})
		targetGroupVersions[groupPackageName] = groupVersionsEntry

		if len(typesWithInformers) != 0 {
			orderer := namer.Orderer{Namer: namer.NewPrivateNamer(0)}
			typesWithInformers = orderer.OrderTypes(typesWithInformers)

			// Generate the informer and fake, for each type.
			packageList = append(packageList, versionInformerPackages(versionPackagePath, groupPackageName, gv, groupGoNames[groupPackageName], boilerplate, typesWithInformers, customArgs)...)
		}
	}

	return packageList
}

func vendorless(p string) string {
	if pos := strings.LastIndex(p, "/vendor/"); pos != -1 {
		return p[pos+len("/vendor/"):]
	}
	return p
}

func typedInformerPackage(groupPkgName string, gv clientgentypes.GroupVersion, externalVersionsInformersPackage string) string {
	return filepath.Join(externalVersionsInformersPackage, groupPkgName, gv.Version.String())
}

func versionClientsPackages(basePackage string, boilerplate []byte, customArgs *informergenargs.CustomArgs) []generator.Package {
	packagePath := filepath.Join(basePackage, "client")

	return []generator.Package{
		// Impl
		&generator.DefaultPackage{
			PackageName: "client",
			PackagePath: packagePath,
			HeaderText:  boilerplate,
			GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
				// Impl
				generators = append(generators, &clientGenerator{
					DefaultGen: generator.DefaultGen{
						OptionalName: "client",
					},
					outputPackage:    packagePath,
					imports:          generator.NewImportTracker(),
					clientSetPackage: customArgs.VersionedClientSetPackage,
				})
				return generators
			},
			FilterFunc: func(c *generator.Context, t *types.Type) bool {
				tags := MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
				return tags.NeedsInformerInjection()
			},
		},
		// Fake
		&generator.DefaultPackage{
			PackageName: "fake",
			PackagePath: filepath.Join(packagePath, "fake"),
			HeaderText:  boilerplate,
			GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
				// Impl
				generators = append(generators, &fakeClientGenerator{
					DefaultGen: generator.DefaultGen{
						OptionalName: "fake",
					},
					outputPackage:      filepath.Join(packagePath, "fake"),
					imports:            generator.NewImportTracker(),
					fakeClientPkg:      filepath.Join(customArgs.VersionedClientSetPackage, "fake"),
					clientInjectionPkg: packagePath,
				})
				return generators
			},
			FilterFunc: func(c *generator.Context, t *types.Type) bool {
				tags := MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
				return tags.NeedsInformerInjection()
			},
		},
	}
}

func versionFactoryPackages(basePackage string, boilerplate []byte, customArgs *informergenargs.CustomArgs) []generator.Package {
	packagePath := filepath.Join(basePackage, "informers", "factory")

	return []generator.Package{
		// Impl
		&generator.DefaultPackage{
			PackageName: "factory",
			PackagePath: packagePath,
			HeaderText:  boilerplate,
			GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
				// Impl
				generators = append(generators, &factoryGenerator{
					DefaultGen: generator.DefaultGen{
						OptionalName: "factory",
					},
					outputPackage:                packagePath,
					cachingClientSetPackage:      filepath.Join(basePackage, "client"),
					sharedInformerFactoryPackage: customArgs.ExternalVersionsInformersPackage,
					imports:                      generator.NewImportTracker(),
				})
				return generators
			},
			FilterFunc: func(c *generator.Context, t *types.Type) bool {
				tags := MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
				return tags.NeedsInformerInjection()
			},
		},
		// Fake
		&generator.DefaultPackage{
			PackageName: "fake",
			PackagePath: filepath.Join(packagePath, "fake"),
			HeaderText:  boilerplate,
			GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
				// Impl
				generators = append(generators, &fakeFactoryGenerator{
					DefaultGen: generator.DefaultGen{
						OptionalName: "fake",
					},
					outputPackage:                filepath.Join(packagePath, "fake"),
					factoryInjectionPkg:          packagePath,
					fakeClientInjectionPkg:       filepath.Join(basePackage, "client", "fake"),
					sharedInformerFactoryPackage: customArgs.ExternalVersionsInformersPackage,
					imports:                      generator.NewImportTracker(),
				})
				return generators
			},
			FilterFunc: func(c *generator.Context, t *types.Type) bool {
				tags := MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
				return tags.NeedsInformerInjection()
			},
		},
	}
}

func versionInformerPackages(basePackage string, groupPkgName string, gv clientgentypes.GroupVersion, groupGoName string, boilerplate []byte, typesToGenerate []*types.Type, customArgs *informergenargs.CustomArgs) []generator.Package {
	factoryPackagePath := filepath.Join(basePackage, "informers", "factory")
	packagePath := filepath.Join(basePackage, "informers", groupPkgName, strings.ToLower(gv.Version.NonEmpty()))

	vers := make([]generator.Package, 0, 2*len(typesToGenerate))

	for _, t := range typesToGenerate {
		// Fix for golang iterator bug.
		t := t
		packagePath := packagePath + "/" + strings.ToLower(t.Name.Name)
		typedInformerPackage := typedInformerPackage(groupPkgName, gv, customArgs.ExternalVersionsInformersPackage)

		// Impl
		vers = append(vers, &generator.DefaultPackage{
			PackageName: strings.ToLower(t.Name.Name),
			PackagePath: packagePath,
			HeaderText:  boilerplate,
			GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
				// Impl
				generators = append(generators, &injectionGenerator{
					DefaultGen: generator.DefaultGen{
						OptionalName: strings.ToLower(t.Name.Name),
					},
					outputPackage:               packagePath,
					groupVersion:                gv,
					groupGoName:                 groupGoName,
					typeToGenerate:              t,
					imports:                     generator.NewImportTracker(),
					typedInformerPackage:        typedInformerPackage,
					groupInformerFactoryPackage: factoryPackagePath,
				})
				return generators
			},
			FilterFunc: func(c *generator.Context, t *types.Type) bool {
				tags := MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
				return tags.NeedsInformerInjection()
			},
		})

		// Fake
		vers = append(vers, &generator.DefaultPackage{
			PackageName: "fake",
			PackagePath: filepath.Join(packagePath, "fake"),
			HeaderText:  boilerplate,
			GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
				// Impl
				generators = append(generators, &fakeInformerGenerator{
					DefaultGen: generator.DefaultGen{
						OptionalName: "fake",
					},
					outputPackage:           filepath.Join(packagePath, "fake"),
					imports:                 generator.NewImportTracker(),
					typeToGenerate:          t,
					groupVersion:            gv,
					groupGoName:             groupGoName,
					informerInjectionPkg:    packagePath,
					fakeFactoryInjectionPkg: filepath.Join(factoryPackagePath, "fake"),
				})
				return generators
			},
			FilterFunc: func(c *generator.Context, t *types.Type) bool {
				tags := MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
				return tags.NeedsInformerInjection()
			},
		})
	}
	return vers
}
