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
	"fmt"
	"path/filepath"
	"strings"

	clientgenargs "k8s.io/code-generator/cmd/client-gen/args"
	"k8s.io/code-generator/cmd/client-gen/generators/fake"
	"k8s.io/code-generator/cmd/client-gen/generators/scheme"
	"k8s.io/code-generator/cmd/client-gen/generators/util"
	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"
	codegennamer "k8s.io/code-generator/pkg/namer"
	"k8s.io/gengo/v2/args"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"

	"k8s.io/klog/v2"
)

// NameSystems returns the name system used by the generators in this package.
func NameSystems(pluralExceptions map[string]string) namer.NameSystems {
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
		"resource":           codegennamer.NewTagOverrideNamer("resourceName", lowercaseNamer),
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

func targetForGroup(gv clientgentypes.GroupVersion, typeList []*types.Type, clientsetDir, clientsetPkg string, groupPkgName string, groupGoName string, apiPath string, inputPkg string, applyBuilderPkg string, boilerplate []byte) generator.Target {
	subdir := filepath.Join("typed", strings.ToLower(groupPkgName), strings.ToLower(gv.Version.NonEmpty()))
	gvDir := filepath.Join(clientsetDir, subdir)
	gvPkg := filepath.Join(clientsetPkg, subdir)

	return &generator.SimpleTarget{
		PkgName:       strings.ToLower(gv.Version.NonEmpty()),
		PkgPath:       gvPkg,
		PkgDir:        gvDir,
		HeaderComment: boilerplate,
		PkgDocComment: []byte("// This package has the automatically generated typed clients.\n"),
		// GeneratorsFunc returns a list of generators. Each generator makes a
		// single file.
		GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = []generator.Generator{
				// Always generate a "doc.go" file.
				generator.GoGenerator{OutputFilename: "doc.go"},
			}
			// Since we want a file per type that we generate a client for, we
			// have to provide a function for this.
			for _, t := range typeList {
				generators = append(generators, &genClientForType{
					GoGenerator: generator.GoGenerator{
						OutputFilename: strings.ToLower(c.Namers["private"].Name(t)) + ".go",
					},
					outputPackage:             gvPkg,
					inputPackage:              inputPkg,
					clientsetPackage:          clientsetPkg,
					applyConfigurationPackage: applyBuilderPkg,
					group:                     gv.Group.NonEmpty(),
					version:                   gv.Version.String(),
					groupGoName:               groupGoName,
					typeToMatch:               t,
					imports:                   generator.NewImportTracker(),
				})
			}

			generators = append(generators, &genGroup{
				GoGenerator: generator.GoGenerator{
					OutputFilename: groupPkgName + "_client.go",
				},
				outputPackage:    gvPkg,
				inputPackage:     inputPkg,
				clientsetPackage: clientsetPkg,
				group:            gv.Group.NonEmpty(),
				version:          gv.Version.String(),
				groupGoName:      groupGoName,
				apiPath:          apiPath,
				types:            typeList,
				imports:          generator.NewImportTracker(),
			})

			expansionFileName := "generated_expansion.go"
			generators = append(generators, &genExpansion{
				groupPackagePath: gvDir,
				GoGenerator: generator.GoGenerator{
					OutputFilename: expansionFileName,
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

func targetForClientset(customArgs *clientgenargs.CustomArgs, clientsetDir, clientsetPkg string, groupGoNames map[clientgentypes.GroupVersion]string, boilerplate []byte) generator.Target {
	return &generator.SimpleTarget{
		PkgName:       customArgs.ClientsetName,
		PkgPath:       clientsetPkg,
		PkgDir:        clientsetDir,
		HeaderComment: boilerplate,
		// GeneratorsFunc returns a list of generators. Each generator generates a
		// single file.
		GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = []generator.Generator{
				&genClientset{
					GoGenerator: generator.GoGenerator{
						OutputFilename: "clientset.go",
					},
					groups:           customArgs.Groups,
					groupGoNames:     groupGoNames,
					clientsetPackage: clientsetPkg,
					imports:          generator.NewImportTracker(),
				},
			}
			return generators
		},
	}
}

func targetForScheme(customArgs *clientgenargs.CustomArgs, clientsetDir, clientsetPkg string, groupGoNames map[clientgentypes.GroupVersion]string, boilerplate []byte) generator.Target {
	schemeDir := filepath.Join(clientsetDir, "scheme")
	schemePkg := filepath.Join(clientsetPkg, "scheme")

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

	return &generator.SimpleTarget{
		PkgName:       "scheme",
		PkgPath:       schemePkg,
		PkgDir:        schemeDir,
		HeaderComment: boilerplate,
		PkgDocComment: []byte("// This package contains the scheme of the automatically generated clientset.\n"),
		// GeneratorsFunc returns a list of generators. Each generator generates a
		// single file.
		GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = []generator.Generator{
				// Always generate a "doc.go" file.
				generator.GoGenerator{OutputFilename: "doc.go"},

				&scheme.GenScheme{
					GoGenerator: generator.GoGenerator{
						OutputFilename: "register.go",
					},
					InputPackages:  customArgs.GroupVersionPackages(),
					OutputPackage:  schemePkg,
					OutputPath:     schemeDir,
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

// Because we try to assemble inputs from an input-base and a set of
// group-version arguments, sometimes that comes in as a filesystem path.  This
// function rewrites them all as their canonical Go import-paths.
//
// TODO: Change this tool to just take inputs as Go "patterns" like every other
// gengo tool, then extract GVs from those.
func sanitizePackagePaths(context *generator.Context, ca *clientgenargs.CustomArgs) error {
	for i := range ca.Groups {
		pkg := &ca.Groups[i]
		for j := range pkg.Versions {
			ver := &pkg.Versions[j]
			input := ver.Package
			p := context.Universe[input]
			if p == nil || p.Name == "" {
				pkgs, err := context.FindPackages(input)
				if err != nil {
					return fmt.Errorf("can't find input package %q: %w", input, err)
				}
				p = context.Universe[pkgs[0]]
				if p == nil {
					return fmt.Errorf("can't find input package %q in universe", input)
				}
				ver.Package = p.Path
			}
		}
	}
	return nil
}

// GetTargets makes the client target definition.
func GetTargets(context *generator.Context, arguments *args.GeneratorArgs) []generator.Target {
	boilerplate, err := arguments.LoadGoBoilerplate()
	if err != nil {
		klog.Fatalf("Failed loading boilerplate: %v", err)
	}

	customArgs, ok := arguments.CustomArgs.(*clientgenargs.CustomArgs)
	if !ok {
		klog.Fatalf("cannot convert arguments.CustomArgs to clientgenargs.CustomArgs")
	}
	includedTypesOverrides := customArgs.IncludedTypesOverrides

	if err := sanitizePackagePaths(context, customArgs); err != nil {
		klog.Fatalf("cannot sanitize inputs: %v", err)
	}
	applyGroupOverrides(context.Universe, customArgs)

	gvToTypes := map[clientgentypes.GroupVersion][]*types.Type{}
	groupGoNames := make(map[clientgentypes.GroupVersion]string)
	for gv, inputDir := range customArgs.GroupVersionPackages() {
		p := context.Universe.Package(inputDir)

		// If there's a comment of the form "// +groupGoName=SomeUniqueShortName", use that as
		// the Go group identifier in CamelCase. It defaults
		groupGoNames[gv] = namer.IC(strings.Split(gv.Group.NonEmpty(), ".")[0])
		if override := types.ExtractCommentTags("+", p.Comments)["groupGoName"]; override != nil {
			groupGoNames[gv] = namer.IC(override[0])
		}

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
				// filter out types which don't have genclient.
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

	clientsetDir := filepath.Join(arguments.OutputBase, customArgs.ClientsetName)
	clientsetPkg := filepath.Join(customArgs.OutputPackage, customArgs.ClientsetName)

	var targetList []generator.Target

	targetList = append(targetList,
		targetForClientset(customArgs, clientsetDir, clientsetPkg, groupGoNames, boilerplate))
	targetList = append(targetList,
		targetForScheme(customArgs, clientsetDir, clientsetPkg, groupGoNames, boilerplate))
	if customArgs.FakeClient {
		targetList = append(targetList,
			fake.TargetForClientset(customArgs, clientsetDir, clientsetPkg, groupGoNames, boilerplate))
	}

	// If --clientset-only=true, we don't regenerate the individual typed clients.
	if customArgs.ClientsetOnly {
		return []generator.Target(targetList)
	}

	orderer := namer.Orderer{Namer: namer.NewPrivateNamer(0)}
	gvPackages := customArgs.GroupVersionPackages()
	for _, group := range customArgs.Groups {
		for _, version := range group.Versions {
			gv := clientgentypes.GroupVersion{Group: group.Group, Version: version.Version}
			types := gvToTypes[gv]
			inputPath := gvPackages[gv]
			targetList = append(targetList,
				targetForGroup(
					gv, orderer.OrderTypes(types), clientsetDir, clientsetPkg,
					group.PackageName, groupGoNames[gv], customArgs.ClientsetAPIPath,
					inputPath, customArgs.ApplyConfigurationPackage, boilerplate))
			if customArgs.FakeClient {
				targetList = append(targetList,
					fake.TargetForGroup(gv, orderer.OrderTypes(types), clientsetDir, clientsetPkg, group.PackageName, groupGoNames[gv], inputPath, customArgs.ApplyConfigurationPackage, boilerplate))
			}
		}
	}

	return targetList
}
