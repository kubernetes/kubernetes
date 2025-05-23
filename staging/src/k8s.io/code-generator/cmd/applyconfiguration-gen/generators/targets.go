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
	"fmt"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"

	"k8s.io/code-generator/cmd/applyconfiguration-gen/args"
	"k8s.io/code-generator/cmd/client-gen/generators/util"
	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"
)

const (
	// ApplyConfigurationTypeSuffix is the suffix of generated apply configuration types.
	ApplyConfigurationTypeSuffix = "ApplyConfiguration"
)

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	return namer.NameSystems{
		"public":  namer.NewPublicNamer(0),
		"private": namer.NewPrivateNamer(0),
		"raw":     namer.NewRawNamer("", nil),
	}
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "public"
}

// GetTargets makes the client target definition.
func GetTargets(context *generator.Context, args *args.Args) []generator.Target {
	boilerplate, err := gengo.GoBoilerplate(args.GoHeaderFile, "", gengo.StdGeneratedBy)
	if err != nil {
		klog.Fatalf("Failed loading boilerplate: %v", err)
	}

	pkgTypes := packageTypesForInputs(context, args.OutputPkg)
	initialTypes := args.ExternalApplyConfigurations
	refs := refGraphForReachableTypes(context.Universe, pkgTypes, initialTypes)
	typeModels, err := newTypeModels(args.OpenAPISchemaFilePath, pkgTypes)
	if err != nil {
		klog.Fatalf("Failed build type models from typeModels %s: %v", args.OpenAPISchemaFilePath, err)
	}

	groupVersions := make(map[string]clientgentypes.GroupVersions)
	groupGoNames := make(map[string]string)
	applyConfigsForGroupVersion := make(map[clientgentypes.GroupVersion][]applyConfig)

	var targetList []generator.Target
	for pkg, p := range pkgTypes {
		gv := groupVersion(p)

		var toGenerate []applyConfig
		for _, t := range p.Types {
			// If we don't have an ObjectMeta field, we lack the information required to make the Apply or ApplyStatus call
			// to the kube-apiserver, so we don't need to generate the type at all
			clientTags := genclientTags(t)
			if clientTags.GenerateClient && !hasObjectMetaField(t) {
				klog.V(5).Infof("skipping type %v because does not have ObjectMeta", t)
				continue
			}
			gvk := gv.WithKind(clientgentypes.Kind(t.Name.Name))
			openAPIName := typeModels.gvkToOpenAPIType[gvk]

			if typePkg, ok := refs[t.Name]; ok {
				toGenerate = append(toGenerate, applyConfig{
					Type:               t,
					ApplyConfiguration: types.Ref(typePkg, t.Name.Name+ApplyConfigurationTypeSuffix),
					OpenAPIName:        openAPIName,
				})
			}
		}
		if len(toGenerate) == 0 {
			continue // Don't generate empty packages
		}
		sort.Sort(applyConfigSort(toGenerate))

		// Apparently we allow the groupName to be overridden in a way that it
		// no longer maps to a Go package by name.  So we have to figure out
		// the offset of this particular output package (pkg) from the base
		// output package (args.OutputPkg).
		pkgSubdir := strings.TrimPrefix(pkg, args.OutputPkg+"/")

		// generate the apply configurations
		targetList = append(targetList,
			targetForApplyConfigurationsPackage(
				args.OutputDir, args.OutputPkg, pkgSubdir,
				boilerplate, gv, toGenerate, refs, typeModels))

		// group all the generated apply configurations by gv so ForKind() can be generated
		groupPackageName := gv.Group.NonEmpty()
		groupVersionsEntry, ok := groupVersions[groupPackageName]
		if !ok {
			groupVersionsEntry = clientgentypes.GroupVersions{
				PackageName: groupPackageName,
				Group:       gv.Group,
			}
		}
		groupVersionsEntry.Versions = append(groupVersionsEntry.Versions, clientgentypes.PackageVersion{
			Version: gv.Version,
			Package: path.Clean(p.Path),
		})

		groupGoNames[groupPackageName] = goName(gv, p)
		applyConfigsForGroupVersion[gv] = toGenerate
		groupVersions[groupPackageName] = groupVersionsEntry
	}

	// generate ForKind() utility function
	targetList = append(targetList,
		targetForUtils(args.OutputDir, args.OutputPkg,
			boilerplate, groupVersions, applyConfigsForGroupVersion, groupGoNames, typeModels))
	// generate internal embedded schema, required for generated Extract functions
	targetList = append(targetList,
		targetForInternal(args.OutputDir, args.OutputPkg,
			boilerplate, typeModels))

	return targetList
}

func typeName(t *types.Type) string {
	typePackage := t.Name.Package
	return fmt.Sprintf("%s.%s", typePackage, t.Name.Name)
}

func targetForApplyConfigurationsPackage(outputDirBase, outputPkgBase, pkgSubdir string, boilerplate []byte, gv clientgentypes.GroupVersion, typesToGenerate []applyConfig, refs refGraph, models *typeModels) generator.Target {
	outputDir := filepath.Join(outputDirBase, pkgSubdir)
	outputPkg := path.Join(outputPkgBase, pkgSubdir)

	return &generator.SimpleTarget{
		PkgName:       gv.Version.PackageName(),
		PkgPath:       outputPkg,
		PkgDir:        outputDir,
		HeaderComment: boilerplate,
		GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
			for _, toGenerate := range typesToGenerate {
				var openAPIType *string
				gvk := gv.WithKind(clientgentypes.Kind(toGenerate.Type.Name.Name))
				if v, ok := models.gvkToOpenAPIType[gvk]; ok {
					openAPIType = &v
				}

				generators = append(generators, &applyConfigurationGenerator{
					GoGenerator: generator.GoGenerator{
						OutputFilename: strings.ToLower(toGenerate.Type.Name.Name) + ".go",
					},
					outPkgBase:   outputPkgBase,
					localPkg:     outputPkg,
					groupVersion: gv,
					applyConfig:  toGenerate,
					imports:      generator.NewImportTrackerForPackage(outputPkg),
					refGraph:     refs,
					openAPIType:  openAPIType,
				})
			}
			return generators
		},
	}
}

func targetForUtils(outputDirBase, outputPkgBase string, boilerplate []byte, groupVersions map[string]clientgentypes.GroupVersions,
	applyConfigsForGroupVersion map[clientgentypes.GroupVersion][]applyConfig, groupGoNames map[string]string, models *typeModels) generator.Target {
	return &generator.SimpleTarget{
		PkgName:       path.Base(outputPkgBase),
		PkgPath:       outputPkgBase,
		PkgDir:        outputDirBase,
		HeaderComment: boilerplate,
		GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = append(generators, &utilGenerator{
				GoGenerator: generator.GoGenerator{
					OutputFilename: "utils.go",
				},
				outputPackage:        outputPkgBase,
				imports:              generator.NewImportTrackerForPackage(outputPkgBase),
				groupVersions:        groupVersions,
				typesForGroupVersion: applyConfigsForGroupVersion,
				groupGoNames:         groupGoNames,
				typeModels:           models,
			})
			return generators
		},
	}
}

func targetForInternal(outputDirBase, outputPkgBase string, boilerplate []byte, models *typeModels) generator.Target {
	outputDir := filepath.Join(outputDirBase, "internal")
	outputPkg := path.Join(outputPkgBase, "internal")
	return &generator.SimpleTarget{
		PkgName:       path.Base(outputPkg),
		PkgPath:       outputPkg,
		PkgDir:        outputDir,
		HeaderComment: boilerplate,
		GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = append(generators, &internalGenerator{
				GoGenerator: generator.GoGenerator{
					OutputFilename: "internal.go",
				},
				outputPackage: outputPkgBase,
				imports:       generator.NewImportTrackerForPackage(outputPkg),
				typeModels:    models,
			})
			return generators
		},
	}
}

func goName(gv clientgentypes.GroupVersion, p *types.Package) string {
	goName := namer.IC(strings.Split(gv.Group.NonEmpty(), ".")[0])
	if override := gengo.ExtractCommentTags("+", p.Comments)["groupGoName"]; override != nil {
		goName = namer.IC(override[0])
	}
	return goName
}

func packageTypesForInputs(context *generator.Context, outPkgBase string) map[string]*types.Package {
	pkgTypes := map[string]*types.Package{}
	for _, inputDir := range context.Inputs {
		p := context.Universe.Package(inputDir)
		internal := isInternalPackage(p)
		if internal {
			klog.Warningf("Skipping internal package: %s", p.Path)
			continue
		}
		// This is how the client generator finds the package we are creating. It uses the API package name, not the group name.
		// This matches the approach of the client-gen, so the two generator can work together.
		// For example, if openshift/api/cloudnetwork/v1 contains an apigroup cloud.network.openshift.io, the client-gen
		// builds a package called cloudnetwork/v1 to contain it. This change makes the applyconfiguration-gen use the same.
		_, gvPackageString := util.ParsePathGroupVersion(p.Path)
		pkg := path.Join(outPkgBase, strings.ToLower(gvPackageString))
		pkgTypes[pkg] = p
	}
	return pkgTypes
}

func groupVersion(p *types.Package) (gv clientgentypes.GroupVersion) {
	parts := strings.Split(p.Path, "/")
	gv.Group = clientgentypes.Group(parts[len(parts)-2])
	gv.Version = clientgentypes.Version(parts[len(parts)-1])

	// If there's a comment of the form "// +groupName=somegroup" or
	// "// +groupName=somegroup.foo.bar.io", use the first field (somegroup) as the name of the
	// group when generating.
	if override := gengo.ExtractCommentTags("+", p.Comments)["groupName"]; override != nil {
		gv.Group = clientgentypes.Group(override[0])
	}
	return gv
}

// isInternalPackage returns true if the package is an internal package
func isInternalPackage(p *types.Package) bool {
	for _, t := range p.Types {
		for _, member := range t.Members {
			if member.Name == "ObjectMeta" {
				return isInternal(member)
			}
		}
	}
	return false
}

// isInternal returns true if the tags for a member do not contain a json tag
func isInternal(m types.Member) bool {
	_, ok := lookupJSONTags(m)
	return !ok
}

func hasObjectMetaField(t *types.Type) bool {
	for _, member := range t.Members {
		if objectMeta.Name == member.Type.Name && member.Embedded {
			return true
		}
	}
	return false
}
