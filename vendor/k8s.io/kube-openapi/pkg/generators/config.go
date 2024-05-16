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

	"k8s.io/gengo/v2"
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

func GetTargets(context *generator.Context, args *args.Args) []generator.Target {
	boilerplate, err := gengo.GoBoilerplate(args.GoHeaderFile, gengo.StdBuildTag, gengo.StdGeneratedBy)
	if err != nil {
		klog.Fatalf("Failed loading boilerplate: %v", err)
	}

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
