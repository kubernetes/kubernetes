/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"path"
	"path/filepath"
	"strings"

	"k8s.io/gengo/args"
	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
	"k8s.io/klog/v2"

	"k8s.io/code-generator/cmd/client-gen/generators/util"
	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"
	informergenargs "k8s.io/code-generator/cmd/informer-gen/args"
	genutil "k8s.io/code-generator/pkg/util"
)

// NameSystems returns the name system used by the generators in this package.
func NameSystems(pluralExceptions map[string]string) namer.NameSystems {
	return namer.NameSystems{
		"public":             namer.NewPublicNamer(0),
		"private":            namer.NewPrivateNamer(0),
		"raw":                namer.NewRawNamer("", nil),
		"publicPlural":       namer.NewPublicPluralNamer(pluralExceptions),
		"allLowercasePlural": namer.NewAllLowercasePluralNamer(pluralExceptions),
		"lowercaseSingular":  &lowercaseSingularNamer{},
	}
}

// lowercaseSingularNamer implements Namer
type lowercaseSingularNamer struct{}

// Name returns t's name in all lowercase.
func (n *lowercaseSingularNamer) Name(t *types.Type) string {
	return strings.ToLower(t.Name.Name)
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "public"
}

// objectMetaForPackage returns the type of ObjectMeta used by package p.
func objectMetaForPackage(p *types.Package) (*types.Type, bool, error) {
	generatingForPackage := false
	for _, t := range p.Types {
		if !util.MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...)).GenerateClient {
			continue
		}
		generatingForPackage = true
		for _, member := range t.Members {
			if member.Name == "ObjectMeta" {
				return member.Type, isInternal(member), nil
			}
		}
	}
	if generatingForPackage {
		return nil, false, fmt.Errorf("unable to find ObjectMeta for any types in package %s", p.Path)
	}
	return nil, false, nil
}

// isInternal returns true if the tags for a member do not contain a json tag
func isInternal(m types.Member) bool {
	return !strings.Contains(m.Tags, "json")
}

func packageForInternalInterfaces(base string) string {
	return filepath.Join(base, "internalinterfaces")
}

func vendorless(p string) string {
	if pos := strings.LastIndex(p, "/vendor/"); pos != -1 {
		return p[pos+len("/vendor/"):]
	}
	return p
}

// Packages makes the client package definition.
func Packages(context *generator.Context, arguments *args.GeneratorArgs) generator.Packages {
	boilerplate, err := arguments.LoadGoBoilerplate()
	if err != nil {
		klog.Fatalf("Failed loading boilerplate: %v", err)
	}

	customArgs, ok := arguments.CustomArgs.(*informergenargs.CustomArgs)
	if !ok {
		klog.Fatalf("Wrong CustomArgs type: %T", arguments.CustomArgs)
	}

	internalVersionPackagePath := filepath.Join(arguments.OutputPackagePath)
	externalVersionPackagePath := filepath.Join(arguments.OutputPackagePath)
	if !customArgs.SingleDirectory {
		internalVersionPackagePath = filepath.Join(arguments.OutputPackagePath, "internalversion")
		externalVersionPackagePath = filepath.Join(arguments.OutputPackagePath, "externalversions")
	}

	var packageList generator.Packages
	typesForGroupVersion := make(map[clientgentypes.GroupVersion][]*types.Type)

	externalGroupVersions := make(map[string]clientgentypes.GroupVersions)
	internalGroupVersions := make(map[string]clientgentypes.GroupVersions)
	groupGoNames := make(map[string]string)
	for _, inputDir := range arguments.InputDirs {
		p := context.Universe.Package(vendorless(inputDir))

		objectMeta, internal, err := objectMetaForPackage(p)
		if err != nil {
			klog.Fatal(err)
		}
		if objectMeta == nil {
			// no types in this package had genclient
			continue
		}

		var gv clientgentypes.GroupVersion
		var targetGroupVersions map[string]clientgentypes.GroupVersions

		if internal {
			lastSlash := strings.LastIndex(p.Path, "/")
			if lastSlash == -1 {
				klog.Fatalf("error constructing internal group version for package %q", p.Path)
			}
			gv.Group = clientgentypes.Group(p.Path[lastSlash+1:])
			targetGroupVersions = internalGroupVersions
		} else {
			parts := strings.Split(p.Path, "/")
			gv.Group = clientgentypes.Group(parts[len(parts)-2])
			gv.Version = clientgentypes.Version(parts[len(parts)-1])
			targetGroupVersions = externalGroupVersions
		}
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
		groupGoNames[groupPackageName] = namer.IC(strings.Split(gv.Group.NonEmpty(), ".")[0])
		if override := types.ExtractCommentTags("+", p.Comments)["groupGoName"]; override != nil {
			groupGoNames[groupPackageName] = namer.IC(override[0])
		}

		var typesToGenerate []*types.Type
		for _, t := range p.Types {
			tags := util.MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
			if !tags.GenerateClient || tags.NoVerbs || !tags.HasVerb("list") || !tags.HasVerb("watch") {
				continue
			}

			typesToGenerate = append(typesToGenerate, t)

			if _, ok := typesForGroupVersion[gv]; !ok {
				typesForGroupVersion[gv] = []*types.Type{}
			}
			typesForGroupVersion[gv] = append(typesForGroupVersion[gv], t)
		}
		if len(typesToGenerate) == 0 {
			continue
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

		orderer := namer.Orderer{Namer: namer.NewPrivateNamer(0)}
		typesToGenerate = orderer.OrderTypes(typesToGenerate)

		if internal {
			packageList = append(packageList, versionPackage(internalVersionPackagePath, groupPackageName, gv, groupGoNames[groupPackageName], boilerplate, typesToGenerate, customArgs.InternalClientSetPackage, customArgs.ListersPackage))
		} else {
			packageList = append(packageList, versionPackage(externalVersionPackagePath, groupPackageName, gv, groupGoNames[groupPackageName], boilerplate, typesToGenerate, customArgs.VersionedClientSetPackage, customArgs.ListersPackage))
		}
	}

	if len(externalGroupVersions) != 0 {
		packageList = append(packageList, factoryInterfacePackage(externalVersionPackagePath, boilerplate, customArgs.VersionedClientSetPackage))
		packageList = append(packageList, factoryPackage(externalVersionPackagePath, boilerplate, groupGoNames, genutil.PluralExceptionListToMapOrDie(customArgs.PluralExceptions), externalGroupVersions,
			customArgs.VersionedClientSetPackage,
			typesForGroupVersion))
		for _, gvs := range externalGroupVersions {
			packageList = append(packageList, groupPackage(externalVersionPackagePath, gvs, boilerplate))
		}
	}

	if len(internalGroupVersions) != 0 {
		packageList = append(packageList, factoryInterfacePackage(internalVersionPackagePath, boilerplate, customArgs.InternalClientSetPackage))
		packageList = append(packageList, factoryPackage(internalVersionPackagePath, boilerplate, groupGoNames, genutil.PluralExceptionListToMapOrDie(customArgs.PluralExceptions), internalGroupVersions, customArgs.InternalClientSetPackage, typesForGroupVersion))
		for _, gvs := range internalGroupVersions {
			packageList = append(packageList, groupPackage(internalVersionPackagePath, gvs, boilerplate))
		}
	}

	return packageList
}

func factoryPackage(basePackage string, boilerplate []byte, groupGoNames, pluralExceptions map[string]string, groupVersions map[string]clientgentypes.GroupVersions, clientSetPackage string,
	typesForGroupVersion map[clientgentypes.GroupVersion][]*types.Type) generator.Package {
	return &generator.DefaultPackage{
		PackageName: filepath.Base(basePackage),
		PackagePath: basePackage,
		HeaderText:  boilerplate,
		GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = append(generators, &factoryGenerator{
				DefaultGen: generator.DefaultGen{
					OptionalName: "factory",
				},
				outputPackage:             basePackage,
				imports:                   generator.NewImportTracker(),
				groupVersions:             groupVersions,
				clientSetPackage:          clientSetPackage,
				internalInterfacesPackage: packageForInternalInterfaces(basePackage),
				gvGoNames:                 groupGoNames,
			})

			generators = append(generators, &genericGenerator{
				DefaultGen: generator.DefaultGen{
					OptionalName: "generic",
				},
				outputPackage:        basePackage,
				imports:              generator.NewImportTracker(),
				groupVersions:        groupVersions,
				pluralExceptions:     pluralExceptions,
				typesForGroupVersion: typesForGroupVersion,
				groupGoNames:         groupGoNames,
			})

			return generators
		},
	}
}

func factoryInterfacePackage(basePackage string, boilerplate []byte, clientSetPackage string) generator.Package {
	packagePath := packageForInternalInterfaces(basePackage)

	return &generator.DefaultPackage{
		PackageName: filepath.Base(packagePath),
		PackagePath: packagePath,
		HeaderText:  boilerplate,
		GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = append(generators, &factoryInterfaceGenerator{
				DefaultGen: generator.DefaultGen{
					OptionalName: "factory_interfaces",
				},
				outputPackage:    packagePath,
				imports:          generator.NewImportTracker(),
				clientSetPackage: clientSetPackage,
			})

			return generators
		},
	}
}

func groupPackage(basePackage string, groupVersions clientgentypes.GroupVersions, boilerplate []byte) generator.Package {
	packagePath := filepath.Join(basePackage, groupVersions.PackageName)
	groupPkgName := strings.Split(string(groupVersions.PackageName), ".")[0]

	return &generator.DefaultPackage{
		PackageName: groupPkgName,
		PackagePath: packagePath,
		HeaderText:  boilerplate,
		GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = append(generators, &groupInterfaceGenerator{
				DefaultGen: generator.DefaultGen{
					OptionalName: "interface",
				},
				outputPackage:             packagePath,
				groupVersions:             groupVersions,
				imports:                   generator.NewImportTracker(),
				internalInterfacesPackage: packageForInternalInterfaces(basePackage),
			})
			return generators
		},
		FilterFunc: func(c *generator.Context, t *types.Type) bool {
			tags := util.MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
			return tags.GenerateClient && tags.HasVerb("list") && tags.HasVerb("watch")
		},
	}
}

func versionPackage(basePackage string, groupPkgName string, gv clientgentypes.GroupVersion, groupGoName string, boilerplate []byte, typesToGenerate []*types.Type, clientSetPackage, listersPackage string) generator.Package {
	packagePath := filepath.Join(basePackage, groupPkgName, strings.ToLower(gv.Version.NonEmpty()))

	return &generator.DefaultPackage{
		PackageName: strings.ToLower(gv.Version.NonEmpty()),
		PackagePath: packagePath,
		HeaderText:  boilerplate,
		GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = append(generators, &versionInterfaceGenerator{
				DefaultGen: generator.DefaultGen{
					OptionalName: "interface",
				},
				outputPackage:             packagePath,
				imports:                   generator.NewImportTracker(),
				types:                     typesToGenerate,
				internalInterfacesPackage: packageForInternalInterfaces(basePackage),
			})

			for _, t := range typesToGenerate {
				generators = append(generators, &informerGenerator{
					DefaultGen: generator.DefaultGen{
						OptionalName: strings.ToLower(t.Name.Name),
					},
					outputPackage:             packagePath,
					groupPkgName:              groupPkgName,
					groupVersion:              gv,
					groupGoName:               groupGoName,
					typeToGenerate:            t,
					imports:                   generator.NewImportTracker(),
					clientSetPackage:          clientSetPackage,
					listersPackage:            listersPackage,
					internalInterfacesPackage: packageForInternalInterfaces(basePackage),
				})
			}
			return generators
		},
		FilterFunc: func(c *generator.Context, t *types.Type) bool {
			tags := util.MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
			return tags.GenerateClient && tags.HasVerb("list") && tags.HasVerb("watch")
		},
	}
}
