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

// Package generators has the generators for the client-gen utility.
package generators

import (
	"path/filepath"
	"strings"

	clientgenargs "k8s.io/code-generator/cmd/client-gen/args"
	"k8s.io/code-generator/cmd/client-gen/generators/fake"
	"k8s.io/code-generator/cmd/client-gen/generators/scheme"
	"k8s.io/code-generator/cmd/client-gen/generators/util"
	"k8s.io/code-generator/cmd/client-gen/path"
	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"
	"k8s.io/gengo/args"
	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"

	"github.com/golang/glog"
)

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	pluralExceptions := map[string]string{
		"Endpoints": "Endpoints",
	}
	lowercaseNamer := namer.NewAllLowercasePluralNamer(pluralExceptions)

	publicNamer := &ExceptionNamer{
		Exceptions: map[string]string{
			// these exceptions are used to deconflict the generated code
			// you can put your fully qualified package like
			// to generate a name that doesn't conflict with your group.
			// "k8s.io/apis/events/v1beta1.Event": "EventResource"
		},
		KeyFunc: func(t *types.Type) string {
			return t.Name.Package + "." + t.Name.Name
		},
		Delegate: namer.NewPublicNamer(0),
	}
	privateNamer := &ExceptionNamer{
		Exceptions: map[string]string{
			// these exceptions are used to deconflict the generated code
			// you can put your fully qualified package like
			// to generate a name that doesn't conflict with your group.
			// "k8s.io/apis/events/v1beta1.Event": "eventResource"
		},
		KeyFunc: func(t *types.Type) string {
			return t.Name.Package + "." + t.Name.Name
		},
		Delegate: namer.NewPrivateNamer(0),
	}
	publicPluralNamer := &ExceptionNamer{
		Exceptions: map[string]string{
			// these exceptions are used to deconflict the generated code
			// you can put your fully qualified package like
			// to generate a name that doesn't conflict with your group.
			// "k8s.io/apis/events/v1beta1.Event": "EventResource"
		},
		KeyFunc: func(t *types.Type) string {
			return t.Name.Package + "." + t.Name.Name
		},
		Delegate: namer.NewPublicPluralNamer(pluralExceptions),
	}
	privatePluralNamer := &ExceptionNamer{
		Exceptions: map[string]string{
			// you can put your fully qualified package like
			// to generate a name that doesn't conflict with your group.
			// "k8s.io/apis/events/v1beta1.Event": "eventResource"
			// these exceptions are used to deconflict the generated code
			"k8s.io/apis/events/v1beta1.Event":        "eventResources",
			"k8s.io/kubernetes/pkg/apis/events.Event": "eventResources",
		},
		KeyFunc: func(t *types.Type) string {
			return t.Name.Package + "." + t.Name.Name
		},
		Delegate: namer.NewPrivatePluralNamer(pluralExceptions),
	}

	return namer.NameSystems{
		"singularKind":       namer.NewPublicNamer(0),
		"public":             publicNamer,
		"private":            privateNamer,
		"raw":                namer.NewRawNamer("", nil),
		"publicPlural":       publicPluralNamer,
		"privatePlural":      privatePluralNamer,
		"allLowercasePlural": lowercaseNamer,
		"resource":           NewTagOverrideNamer("resourceName", lowercaseNamer),
	}
}

// ExceptionNamer allows you specify exceptional cases with exact names.  This allows you to have control
// for handling various conflicts, like group and resource names for instance.
type ExceptionNamer struct {
	Exceptions map[string]string
	KeyFunc    func(*types.Type) string

	Delegate namer.Namer
}

// Name provides the requested name for a type.
func (n *ExceptionNamer) Name(t *types.Type) string {
	key := n.KeyFunc(t)
	if exception, ok := n.Exceptions[key]; ok {
		return exception
	}
	return n.Delegate.Name(t)
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "public"
}

func packageForGroup(gv clientgentypes.GroupVersion, typeList []*types.Type, clientsetPackage string, groupPackageName string, groupGoName string, apiPath string, srcTreePath string, inputPackage string, boilerplate []byte) generator.Package {
	groupVersionClientPackage := strings.ToLower(filepath.Join(clientsetPackage, "typed", groupPackageName, gv.Version.NonEmpty()))
	return &generator.DefaultPackage{
		PackageName: strings.ToLower(gv.Version.NonEmpty()),
		PackagePath: groupVersionClientPackage,
		HeaderText:  boilerplate,
		PackageDocumentation: []byte(
			`// This package has the automatically generated typed clients.
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
					outputPackage:    groupVersionClientPackage,
					clientsetPackage: clientsetPackage,
					group:            gv.Group.NonEmpty(),
					version:          gv.Version.String(),
					groupGoName:      groupGoName,
					typeToMatch:      t,
					imports:          generator.NewImportTracker(),
				})
			}

			generators = append(generators, &genGroup{
				DefaultGen: generator.DefaultGen{
					OptionalName: groupPackageName + "_client",
				},
				outputPackage:    groupVersionClientPackage,
				inputPackage:     inputPackage,
				clientsetPackage: clientsetPackage,
				group:            gv.Group.NonEmpty(),
				version:          gv.Version.String(),
				groupGoName:      groupGoName,
				apiPath:          apiPath,
				types:            typeList,
				imports:          generator.NewImportTracker(),
			})

			expansionFileName := "generated_expansion"
			generators = append(generators, &genExpansion{
				groupPackagePath: filepath.Join(srcTreePath, groupVersionClientPackage),
				DefaultGen: generator.DefaultGen{
					OptionalName: expansionFileName,
				},
				types: typeList,
			})

			return generators
		},
		FilterFunc: func(c *generator.Context, t *types.Type) bool {
			return util.MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...)).GenerateClient
		},
	}
}

func packageForClientset(customArgs *clientgenargs.CustomArgs, clientsetPackage string, groupGoNames map[clientgentypes.GroupVersion]string, boilerplate []byte) generator.Package {
	return &generator.DefaultPackage{
		PackageName: customArgs.ClientsetName,
		PackagePath: clientsetPackage,
		HeaderText:  boilerplate,
		PackageDocumentation: []byte(
			`// This package has the automatically generated clientset.
`),
		// GeneratorFunc returns a list of generators. Each generator generates a
		// single file.
		GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = []generator.Generator{
				// Always generate a "doc.go" file.
				generator.DefaultGen{OptionalName: "doc"},

				&genClientset{
					DefaultGen: generator.DefaultGen{
						OptionalName: "clientset",
					},
					groups:           customArgs.Groups,
					groupGoNames:     groupGoNames,
					clientsetPackage: clientsetPackage,
					outputPackage:    customArgs.ClientsetName,
					imports:          generator.NewImportTracker(),
				},
			}
			return generators
		},
	}
}

func packageForScheme(customArgs *clientgenargs.CustomArgs, clientsetPackage string, srcTreePath string, groupGoNames map[clientgentypes.GroupVersion]string, boilerplate []byte) generator.Package {
	schemePackage := filepath.Join(clientsetPackage, "scheme")

	// create runtime.Registry for internal client because it has to know about group versions
	internalClient := false
NextGroup:
	for _, group := range customArgs.Groups {
		for _, v := range group.Versions {
			if v.String() == "" {
				internalClient = true
				break NextGroup
			}
		}
	}

	return &generator.DefaultPackage{
		PackageName: "scheme",
		PackagePath: schemePackage,
		HeaderText:  boilerplate,
		PackageDocumentation: []byte(
			`// This package contains the scheme of the automatically generated clientset.
`),
		// GeneratorFunc returns a list of generators. Each generator generates a
		// single file.
		GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = []generator.Generator{
				// Always generate a "doc.go" file.
				generator.DefaultGen{OptionalName: "doc"},

				&scheme.GenScheme{
					DefaultGen: generator.DefaultGen{
						OptionalName: "register",
					},
					InputPackages:  customArgs.GroupVersionPackages(),
					OutputPackage:  schemePackage,
					OutputPath:     filepath.Join(srcTreePath, schemePackage),
					Groups:         customArgs.Groups,
					GroupGoNames:   groupGoNames,
					ImportTracker:  generator.NewImportTracker(),
					CreateRegistry: internalClient,
				},
			}
			return generators
		},
	}
}

// applyGroupOverrides applies group name overrides to each package, if applicable. If there is a
// comment of the form "// +groupName=somegroup" or "// +groupName=somegroup.foo.bar.io", use the
// first field (somegroup) as the name of the group in Go code, e.g. as the func name in a clientset.
//
// If the first field of the groupName is not unique within the clientset, use "// +groupName=unique
func applyGroupOverrides(universe types.Universe, customArgs *clientgenargs.CustomArgs) {
	// Create a map from "old GV" to "new GV" so we know what changes we need to make.
	changes := make(map[clientgentypes.GroupVersion]clientgentypes.GroupVersion)
	for gv, inputDir := range customArgs.GroupVersionPackages() {
		p := universe.Package(inputDir)
		if override := types.ExtractCommentTags("+", p.Comments)["groupName"]; override != nil {
			newGV := clientgentypes.GroupVersion{
				Group:   clientgentypes.Group(override[0]),
				Version: gv.Version,
			}
			changes[gv] = newGV
		}
	}

	// Modify customArgs.Groups based on the groupName overrides.
	newGroups := make([]clientgentypes.GroupVersions, 0, len(customArgs.Groups))
	for _, gvs := range customArgs.Groups {
		gv := clientgentypes.GroupVersion{
			Group:   gvs.Group,
			Version: gvs.Versions[0].Version, // we only need a version, and the first will do
		}
		if newGV, ok := changes[gv]; ok {
			// There's an override, so use it.
			newGVS := clientgentypes.GroupVersions{
				PackageName: gvs.PackageName,
				Group:       newGV.Group,
				Versions:    gvs.Versions,
			}
			newGroups = append(newGroups, newGVS)
		} else {
			// No override.
			newGroups = append(newGroups, gvs)
		}
	}
	customArgs.Groups = newGroups
}

// Packages makes the client package definition.
func Packages(context *generator.Context, arguments *args.GeneratorArgs) generator.Packages {
	boilerplate, err := arguments.LoadGoBoilerplate()
	if err != nil {
		glog.Fatalf("Failed loading boilerplate: %v", err)
	}

	customArgs, ok := arguments.CustomArgs.(*clientgenargs.CustomArgs)
	if !ok {
		glog.Fatalf("cannot convert arguments.CustomArgs to clientgenargs.CustomArgs")
	}
	includedTypesOverrides := customArgs.IncludedTypesOverrides

	applyGroupOverrides(context.Universe, customArgs)

	gvToTypes := map[clientgentypes.GroupVersion][]*types.Type{}
	groupGoNames := make(map[clientgentypes.GroupVersion]string)
	for gv, inputDir := range customArgs.GroupVersionPackages() {
		p := context.Universe.Package(path.Vendorless(inputDir))

		// If there's a comment of the form "// +groupGoName=SomeUniqueShortName", use that as
		// the Go group identifier in CamelCase. It defaults
		groupGoNames[gv] = namer.IC(strings.Split(gv.Group.NonEmpty(), ".")[0])
		if override := types.ExtractCommentTags("+", p.Comments)["groupGoName"]; override != nil {
			groupGoNames[gv] = namer.IC(override[0])
		}

		// Package are indexed with the vendor prefix stripped
		for n, t := range p.Types {
			// filter out types which are not included in user specified overrides.
			typesOverride, ok := includedTypesOverrides[gv]
			if ok {
				found := false
				for _, typeStr := range typesOverride {
					if typeStr == n {
						found = true
						break
					}
				}
				if !found {
					continue
				}
			} else {
				// User has not specified any override for this group version.
				// filter out types which dont have genclient.
				if tags := util.MustParseClientGenTags(append(t.SecondClosestCommentLines, t.CommentLines...)); !tags.GenerateClient {
					continue
				}
			}
			if _, found := gvToTypes[gv]; !found {
				gvToTypes[gv] = []*types.Type{}
			}
			gvToTypes[gv] = append(gvToTypes[gv], t)
		}
	}

	var packageList []generator.Package
	clientsetPackage := filepath.Join(arguments.OutputPackagePath, customArgs.ClientsetName)

	packageList = append(packageList, packageForClientset(customArgs, clientsetPackage, groupGoNames, boilerplate))
	packageList = append(packageList, packageForScheme(customArgs, clientsetPackage, arguments.OutputBase, groupGoNames, boilerplate))
	if customArgs.FakeClient {
		packageList = append(packageList, fake.PackageForClientset(customArgs, clientsetPackage, groupGoNames, boilerplate))
	}

	// If --clientset-only=true, we don't regenerate the individual typed clients.
	if customArgs.ClientsetOnly {
		return generator.Packages(packageList)
	}

	orderer := namer.Orderer{Namer: namer.NewPrivateNamer(0)}
	gvPackages := customArgs.GroupVersionPackages()
	for _, group := range customArgs.Groups {
		for _, version := range group.Versions {
			gv := clientgentypes.GroupVersion{Group: group.Group, Version: version.Version}
			types := gvToTypes[gv]
			inputPath := gvPackages[gv]
			packageList = append(packageList, packageForGroup(gv, orderer.OrderTypes(types), clientsetPackage, group.PackageName, groupGoNames[gv], customArgs.ClientsetAPIPath, arguments.OutputBase, inputPath, boilerplate))
			if customArgs.FakeClient {
				packageList = append(packageList, fake.PackageForGroup(gv, orderer.OrderTypes(types), clientsetPackage, group.PackageName, groupGoNames[gv], inputPath, boilerplate))
			}
		}
	}

	return generator.Packages(packageList)
}

// tagOverrideNamer is a namer which pulls names from a given tag, if specified,
// and otherwise falls back to a different namer.
type tagOverrideNamer struct {
	tagName  string
	fallback namer.Namer
}

func (n *tagOverrideNamer) Name(t *types.Type) string {
	if nameOverride := extractTag(n.tagName, append(t.SecondClosestCommentLines, t.CommentLines...)); nameOverride != "" {
		return nameOverride
	}

	return n.fallback.Name(t)
}

// NewTagOverrideNamer creates a namer.Namer which uses the contents of the given tag as
// the name, or falls back to another Namer if the tag is not present.
func NewTagOverrideNamer(tagName string, fallback namer.Namer) namer.Namer {
	return &tagOverrideNamer{
		tagName:  tagName,
		fallback: fallback,
	}
}
