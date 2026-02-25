/*
Copyright 2018 The Kubernetes Authors.

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

	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"
	"k8s.io/kube-openapi/cmd/openapi-gen/args"
)

type identityNamer struct{}

func (_ identityNamer) Name(t *types.Type) string {
	return t.Name.String()
}

var _ namer.Namer = identityNamer{}

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	return namer.NameSystems{
		"raw":           namer.NewRawNamer("", nil),
		"sorting_namer": identityNamer{},
	}
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "sorting_namer"
}

// GetOpenAPITargets returns the targets for OpenAPI definition generation.
func GetOpenAPITargets(context *generator.Context, args *args.Args, boilerplate []byte) []generator.Target {
	reportPath := "-"
	if args.ReportFilename != "" {
		reportPath = args.ReportFilename
	}
	context.FileTypes[apiViolationFileType] = apiViolationFile{
		unmangledPath: reportPath,
	}

	return []generator.Target{
		&generator.SimpleTarget{
			PkgName:       path.Base(args.OutputPkg), // `path` vs. `filepath` because packages use '/'
			PkgPath:       args.OutputPkg,
			PkgDir:        args.OutputDir,
			HeaderComment: boilerplate,
			GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
				return []generator.Generator{
					newOpenAPIGen(
						args.OutputFile,
						args.OutputPkg,
					),
					newAPIViolationGen(),
				}
			},
			FilterFunc: apiTypeFilterFunc,
		},
	}
}

// GetModelNameTargets returns the targets for model name generation.
func GetModelNameTargets(context *generator.Context, args *args.Args, boilerplate []byte) []generator.Target {
	var targets []generator.Target
	for _, i := range context.Inputs {
		klog.V(5).Infof("Considering pkg %q", i)

		pkg := context.Universe[i]

		openAPISchemaNamePackage, err := extractOpenAPISchemaNamePackage(pkg.Comments)
		if err != nil {
			klog.Fatalf("Package %v: invalid %s:%v", i, tagModelPackage, err)
		}
		hasPackageTag := len(openAPISchemaNamePackage) > 0

		hasCandidates := false
		for _, t := range pkg.Types {
			v, err := singularTag(tagModelPackage, t.CommentLines)
			if err != nil {
				klog.Fatalf("Type %v: invalid %s:%v", t.Name, tagModelPackage, err)
			}
			hasTag := hasPackageTag || v != nil
			hasModel := isSchemaNameType(t)
			if hasModel && hasTag {
				hasCandidates = true
				break
			}
		}
		if !hasCandidates {
			klog.V(5).Infof("  skipping package")
			continue
		}

		klog.V(3).Infof("Generating package %q", pkg.Path)

		targets = append(targets,
			&generator.SimpleTarget{
				PkgName:       path.Base(pkg.Path),
				PkgPath:       pkg.Path,
				PkgDir:        pkg.Dir, // output pkg is the same as the input
				HeaderComment: boilerplate,
				FilterFunc: func(c *generator.Context, t *types.Type) bool {
					return t.Name.Package == pkg.Path
				},
				GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
					return []generator.Generator{
						NewSchemaNameGen(args.OutputModelNameFile, pkg.Path, openAPISchemaNamePackage),
					}
				},
			})
	}
	return targets
}
