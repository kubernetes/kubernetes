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

	"k8s.io/code-generator/cmd/client-gen/generators/util"
	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"
	"k8s.io/code-generator/cmd/informer-gen/args"
	genutil "k8s.io/code-generator/pkg/util"
	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"
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

const subdirForInternalInterfaces = "internalinterfaces"

// GetTargets makes the client target definition.
func GetTargets(context *generator.Context, args *args.Args) []generator.Target {
	boilerplate, err := gengo.GoBoilerplate(args.GoHeaderFile, "", gengo.StdGeneratedBy)
	if err != nil {
		klog.Fatalf("Failed loading boilerplate: %v", err)
	}

	internalVersionOutputDir := args.OutputDir
	internalVersionOutputPkg := args.OutputPkg
	externalVersionOutputDir := args.OutputDir
	externalVersionOutputPkg := args.OutputPkg
	if !args.SingleDirectory {
		internalVersionOutputDir = filepath.Join(internalVersionOutputDir, "internalversion")
		internalVersionOutputPkg = path.Join(internalVersionOutputPkg, "internalversion")
		externalVersionOutputDir = filepath.Join(externalVersionOutputDir, "externalversions")
		externalVersionOutputPkg = path.Join(externalVersionOutputPkg, "externalversions")
	}

	var targetList []generator.Target
	typesForGroupVersion := make(map[clientgentypes.GroupVersion][]*types.Type)

	externalGroupVersions := make(map[string]clientgentypes.GroupVersions)
	internalGroupVersions := make(map[string]clientgentypes.GroupVersions)
	groupGoNames := make(map[string]string)
	for _, inputPkg := range context.Inputs {
		p := context.Universe.Package(inputPkg)

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
		if override := gengo.ExtractCommentTags("+", p.Comments)["groupName"]; override != nil {
			gv.Group = clientgentypes.Group(override[0])
		}

		// If there's a comment of the form "// +groupGoName=SomeUniqueShortName", use that as
		// the Go group identifier in CamelCase. It defaults
		groupGoNames[groupPackageName] = namer.IC(strings.Split(gv.Group.NonEmpty(), ".")[0])
		if override := gengo.ExtractCommentTags("+", p.Comments)["groupGoName"]; override != nil {
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
			targetList = append(targetList,
				versionTarget(
					internalVersionOutputDir, internalVersionOutputPkg,
					groupPackageName, gv, groupGoNames[groupPackageName],
					boilerplate, typesToGenerate,
					args.InternalClientSetPackage, args.ListersPackage))
		} else {
			targetList = append(targetList,
				versionTarget(
					externalVersionOutputDir, externalVersionOutputPkg,
					groupPackageName, gv, groupGoNames[groupPackageName],
					boilerplate, typesToGenerate,
					args.VersionedClientSetPackage, args.ListersPackage))
		}
	}

	if len(externalGroupVersions) != 0 {
		targetList = append(targetList,
			factoryInterfaceTarget(
				externalVersionOutputDir, externalVersionOutputPkg,
				boilerplate, args.VersionedClientSetPackage))
		targetList = append(targetList,
			factoryTarget(
				externalVersionOutputDir, externalVersionOutputPkg,
				boilerplate, groupGoNames, genutil.PluralExceptionListToMapOrDie(args.PluralExceptions),
				externalGroupVersions, args.VersionedClientSetPackage, typesForGroupVersion))
		for _, gvs := range externalGroupVersions {
			targetList = append(targetList,
				groupTarget(externalVersionOutputDir, externalVersionOutputPkg, gvs, boilerplate))
		}
	}

	if len(internalGroupVersions) != 0 {
		targetList = append(targetList,
			factoryInterfaceTarget(internalVersionOutputDir, internalVersionOutputPkg, boilerplate, args.InternalClientSetPackage))
		targetList = append(targetList,
			factoryTarget(
				internalVersionOutputDir, internalVersionOutputPkg,
				boilerplate, groupGoNames, genutil.PluralExceptionListToMapOrDie(args.PluralExceptions),
				internalGroupVersions, args.InternalClientSetPackage, typesForGroupVersion))
		for _, gvs := range internalGroupVersions {
			targetList = append(targetList,
				groupTarget(internalVersionOutputDir, internalVersionOutputPkg, gvs, boilerplate))
		}
	}

	return targetList
}

func factoryTarget(outputDirBase, outputPkgBase string, boilerplate []byte, groupGoNames, pluralExceptions map[string]string, groupVersions map[string]clientgentypes.GroupVersions, clientSetPackage string,
	typesForGroupVersion map[clientgentypes.GroupVersion][]*types.Type) generator.Target {
	return &generator.SimpleTarget{
		PkgName:       path.Base(outputDirBase),
		PkgPath:       outputPkgBase,
		PkgDir:        outputDirBase,
		HeaderComment: boilerplate,
		GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = append(generators, &factoryGenerator{
				GoGenerator: generator.GoGenerator{
					OutputFilename: "factory.go",
				},
				outputPackage:             outputPkgBase,
				imports:                   generator.NewImportTrackerForPackage(outputPkgBase),
				groupVersions:             groupVersions,
				clientSetPackage:          clientSetPackage,
				internalInterfacesPackage: path.Join(outputPkgBase, subdirForInternalInterfaces),
				gvGoNames:                 groupGoNames,
			})

			generators = append(generators, &genericGenerator{
				GoGenerator: generator.GoGenerator{
					OutputFilename: "generic.go",
				},
				outputPackage:        outputPkgBase,
				imports:              generator.NewImportTrackerForPackage(outputPkgBase),
				groupVersions:        groupVersions,
				pluralExceptions:     pluralExceptions,
				typesForGroupVersion: typesForGroupVersion,
				groupGoNames:         groupGoNames,
			})

			return generators
		},
	}
}

func factoryInterfaceTarget(outputDirBase, outputPkgBase string, boilerplate []byte, clientSetPackage string) generator.Target {
	outputDir := filepath.Join(outputDirBase, subdirForInternalInterfaces)
	outputPkg := path.Join(outputPkgBase, subdirForInternalInterfaces)

	return &generator.SimpleTarget{
		PkgName:       path.Base(outputDir),
		PkgPath:       outputPkg,
		PkgDir:        outputDir,
		HeaderComment: boilerplate,
		GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = append(generators, &factoryInterfaceGenerator{
				GoGenerator: generator.GoGenerator{
					OutputFilename: "factory_interfaces.go",
				},
				outputPackage:    outputPkg,
				imports:          generator.NewImportTrackerForPackage(outputPkg),
				clientSetPackage: clientSetPackage,
			})

			return generators
		},
	}
}

func groupTarget(outputDirBase, outputPackageBase string, groupVersions clientgentypes.GroupVersions, boilerplate []byte) generator.Target {
	outputDir := filepath.Join(outputDirBase, groupVersions.PackageName)
	outputPkg := path.Join(outputPackageBase, groupVersions.PackageName)
	groupPkgName := strings.Split(string(groupVersions.PackageName), ".")[0]

	return &generator.SimpleTarget{
		PkgName:       groupPkgName,
		PkgPath:       outputPkg,
		PkgDir:        outputDir,
		HeaderComment: boilerplate,
		GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = append(generators, &groupInterfaceGenerator{
				GoGenerator: generator.GoGenerator{
					OutputFilename: "interface.go",
				},
				outputPackage:             outputPkg,
				groupVersions:             groupVersions,
				imports:                   generator.NewImportTrackerForPackage(outputPkg),
				internalInterfacesPackage: path.Join(outputPackageBase, subdirForInternalInterfaces),
			})
			return generators
		},
		FilterFunc: func(c *generator.Context, t *types.Type) bool {
			tags := util.MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...))
			return tags.GenerateClient && tags.HasVerb("list") && tags.HasVerb("watch")
		},
	}
}

func versionTarget(outputDirBase, outputPkgBase string, groupPkgName string, gv clientgentypes.GroupVersion, groupGoName string, boilerplate []byte, typesToGenerate []*types.Type, clientSetPackage, listersPackage string) generator.Target {
	subdir := []string{groupPkgName, strings.ToLower(gv.Version.NonEmpty())}
	outputDir := filepath.Join(outputDirBase, filepath.Join(subdir...))
	outputPkg := path.Join(outputPkgBase, path.Join(subdir...))

	return &generator.SimpleTarget{
		PkgName:       strings.ToLower(gv.Version.NonEmpty()),
		PkgPath:       outputPkg,
		PkgDir:        outputDir,
		HeaderComment: boilerplate,
		GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = append(generators, &versionInterfaceGenerator{
				GoGenerator: generator.GoGenerator{
					OutputFilename: "interface.go",
				},
				outputPackage:             outputPkg,
				imports:                   generator.NewImportTrackerForPackage(outputPkg),
				types:                     typesToGenerate,
				internalInterfacesPackage: path.Join(outputPkgBase, subdirForInternalInterfaces),
			})

			for _, t := range typesToGenerate {
				generators = append(generators, &informerGenerator{
					GoGenerator: generator.GoGenerator{
						OutputFilename: strings.ToLower(t.Name.Name) + ".go",
					},
					outputPackage:             outputPkg,
					groupPkgName:              groupPkgName,
					groupVersion:              gv,
					groupGoName:               groupGoName,
					typeToGenerate:            t,
					imports:                   generator.NewImportTrackerForPackage(outputPkg),
					clientSetPackage:          clientSetPackage,
					listersPackage:            listersPackage,
					internalInterfacesPackage: path.Join(outputPkgBase, subdirForInternalInterfaces),
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
