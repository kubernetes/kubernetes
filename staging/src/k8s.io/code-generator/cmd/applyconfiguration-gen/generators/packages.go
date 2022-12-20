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

	"k8s.io/gengo/args"
	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
	"k8s.io/klog/v2"

	applygenargs "k8s.io/code-generator/cmd/applyconfiguration-gen/args"
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

// Packages makes the client package definition.
func Packages(context *generator.Context, arguments *args.GeneratorArgs) generator.Packages {
	boilerplate, err := arguments.LoadGoBoilerplate()
	if err != nil {
		klog.Fatalf("Failed loading boilerplate: %v", err)
	}

	pkgTypes := packageTypesForInputDirs(context, arguments.InputDirs, arguments.OutputPackagePath)
	customArgs := arguments.CustomArgs.(*applygenargs.CustomArgs)
	initialTypes := customArgs.ExternalApplyConfigurations
	refs := refGraphForReachableTypes(context.Universe, pkgTypes, initialTypes)
	typeModels, err := newTypeModels(customArgs.OpenAPISchemaFilePath, pkgTypes)
	if err != nil {
		klog.Fatalf("Failed build type models from typeModels %s: %v", customArgs.OpenAPISchemaFilePath, err)
	}

	groupVersions := make(map[string]clientgentypes.GroupVersions)
	groupGoNames := make(map[string]string)
	applyConfigsForGroupVersion := make(map[clientgentypes.GroupVersion][]applyConfig)

	var packageList generator.Packages
	for pkg, p := range pkgTypes {
		gv := groupVersion(p)

		pkgType := types.Name{Name: gv.Group.PackageName(), Package: pkg}

		var toGenerate []applyConfig
		for _, t := range p.Types {
			// If we don't have an ObjectMeta field, we lack the information required to make the Apply or ApplyStatus call
			// to the kube-apiserver, so we don't need to generate the type at all
			clientTags := genclientTags(t)
			if clientTags.GenerateClient && !hasObjectMetaField(t) {
				klog.V(5).Infof("skipping type %v because does not have ObjectMeta", t)
				continue
			}
			if typePkg, ok := refs[t.Name]; ok {
				toGenerate = append(toGenerate, applyConfig{
					Type:               t,
					ApplyConfiguration: types.Ref(typePkg, t.Name.Name+ApplyConfigurationTypeSuffix),
				})
			}
		}
		if len(toGenerate) == 0 {
			continue // Don't generate empty packages
		}
		sort.Sort(applyConfigSort(toGenerate))

		// generate the apply configurations
		packageList = append(packageList, generatorForApplyConfigurationsPackage(arguments.OutputPackagePath, boilerplate, pkgType, gv, toGenerate, refs, typeModels))

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
	packageList = append(packageList, generatorForUtils(arguments.OutputPackagePath, boilerplate, groupVersions, applyConfigsForGroupVersion, groupGoNames))
	// generate internal embedded schema, required for generated Extract functions
	packageList = append(packageList, generatorForInternal(filepath.Join(arguments.OutputPackagePath, "internal"), boilerplate, typeModels))

	return packageList
}

func friendlyName(name string) string {
	nameParts := strings.Split(name, "/")
	// Reverse first part. e.g., io.k8s... instead of k8s.io...
	if len(nameParts) > 0 && strings.Contains(nameParts[0], ".") {
		parts := strings.Split(nameParts[0], ".")
		for i, j := 0, len(parts)-1; i < j; i, j = i+1, j-1 {
			parts[i], parts[j] = parts[j], parts[i]
		}
		nameParts[0] = strings.Join(parts, ".")
	}
	return strings.Join(nameParts, ".")
}

func typeName(t *types.Type) string {
	typePackage := t.Name.Package
	if strings.Contains(typePackage, "/vendor/") {
		typePackage = typePackage[strings.Index(typePackage, "/vendor/")+len("/vendor/"):]
	}
	return fmt.Sprintf("%s.%s", typePackage, t.Name.Name)
}

func generatorForApplyConfigurationsPackage(outputPackagePath string, boilerplate []byte, packageName types.Name, gv clientgentypes.GroupVersion, typesToGenerate []applyConfig, refs refGraph, models *typeModels) *generator.DefaultPackage {
	return &generator.DefaultPackage{
		PackageName: gv.Version.PackageName(),
		PackagePath: packageName.Package,
		HeaderText:  boilerplate,
		GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
			for _, toGenerate := range typesToGenerate {
				var openAPIType *string
				gvk := gvk{
					group:   gv.Group.String(),
					version: gv.Version.String(),
					kind:    toGenerate.Type.Name.Name,
				}
				if v, ok := models.gvkToOpenAPIType[gvk]; ok {
					openAPIType = &v
				}

				generators = append(generators, &applyConfigurationGenerator{
					DefaultGen: generator.DefaultGen{
						OptionalName: strings.ToLower(toGenerate.Type.Name.Name),
					},
					outputPackage: outputPackagePath,
					localPackage:  packageName,
					groupVersion:  gv,
					applyConfig:   toGenerate,
					imports:       generator.NewImportTracker(),
					refGraph:      refs,
					openAPIType:   openAPIType,
				})
			}
			return generators
		},
	}
}

func generatorForUtils(outPackagePath string, boilerplate []byte, groupVersions map[string]clientgentypes.GroupVersions, applyConfigsForGroupVersion map[clientgentypes.GroupVersion][]applyConfig, groupGoNames map[string]string) *generator.DefaultPackage {
	return &generator.DefaultPackage{
		PackageName: filepath.Base(outPackagePath),
		PackagePath: outPackagePath,
		HeaderText:  boilerplate,
		GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = append(generators, &utilGenerator{
				DefaultGen: generator.DefaultGen{
					OptionalName: "utils",
				},
				outputPackage:        outPackagePath,
				imports:              generator.NewImportTracker(),
				groupVersions:        groupVersions,
				typesForGroupVersion: applyConfigsForGroupVersion,
				groupGoNames:         groupGoNames,
			})
			return generators
		},
	}
}

func generatorForInternal(outPackagePath string, boilerplate []byte, models *typeModels) *generator.DefaultPackage {
	return &generator.DefaultPackage{
		PackageName: filepath.Base(outPackagePath),
		PackagePath: outPackagePath,
		HeaderText:  boilerplate,
		GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
			generators = append(generators, &internalGenerator{
				DefaultGen: generator.DefaultGen{
					OptionalName: "internal",
				},
				outputPackage: outPackagePath,
				imports:       generator.NewImportTracker(),
				typeModels:    models,
			})
			return generators
		},
	}
}

func goName(gv clientgentypes.GroupVersion, p *types.Package) string {
	goName := namer.IC(strings.Split(gv.Group.NonEmpty(), ".")[0])
	if override := types.ExtractCommentTags("+", p.Comments)["groupGoName"]; override != nil {
		goName = namer.IC(override[0])
	}
	return goName
}

func packageTypesForInputDirs(context *generator.Context, inputDirs []string, outputPath string) map[string]*types.Package {
	pkgTypes := map[string]*types.Package{}
	for _, inputDir := range inputDirs {
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
		pkg := filepath.Join(outputPath, strings.ToLower(gvPackageString))
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
	if override := types.ExtractCommentTags("+", p.Comments)["groupName"]; override != nil {
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
