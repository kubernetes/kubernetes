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
	"fmt"
	"os"
	"path"
	"strings"

	"k8s.io/klog/v2"

	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"
	"k8s.io/code-generator/cmd/register-gen/args"
	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
)

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	return namer.NameSystems{}
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "public"
}

// GetTargets makes targets to generate.
func GetTargets(context *generator.Context, args *args.Args) []generator.Target {
	boilerplate, err := gengo.GoBoilerplate(args.GoHeaderFile, gengo.StdBuildTag, gengo.StdGeneratedBy)
	if err != nil {
		klog.Fatalf("Failed loading boilerplate: %v", err)
	}

	targets := []generator.Target{}
	for _, input := range context.Inputs {
		pkg := context.Universe.Package(input)
		internal, err := isInternal(pkg)
		if err != nil {
			klog.V(5).Infof("skipping the generation of %s file, due to err %v", args.OutputFile, err)
			continue
		}
		if internal {
			klog.V(5).Infof("skipping the generation of %s file because %s package contains internal types, note that internal types don't have \"json\" tags", args.OutputFile, pkg.Name)
			continue
		}
		registerFileName := "register.go"
		searchPath := path.Join(pkg.Dir, registerFileName)
		if _, err := os.Stat(path.Join(searchPath)); err == nil {
			klog.V(5).Infof("skipping the generation of %s file because %s already exists in the path %s", args.OutputFile, registerFileName, searchPath)
			continue
		} else if err != nil && !os.IsNotExist(err) {
			klog.Fatalf("an error %v has occurred while checking if %s exists", err, registerFileName)
		}

		gv := clientgentypes.GroupVersion{}
		{
			pathParts := strings.Split(pkg.Path, "/")
			if len(pathParts) < 2 {
				klog.Errorf("the path of the package must contain the group name and the version, path = %s", pkg.Path)
				continue
			}
			gv.Group = clientgentypes.Group(pathParts[len(pathParts)-2])
			gv.Version = clientgentypes.Version(pathParts[len(pathParts)-1])

			// if there is a comment of the form "// +groupName=somegroup" or "// +groupName=somegroup.foo.bar.io",
			// extract the fully qualified API group name from it and overwrite the group inferred from the package path
			if override := gengo.ExtractCommentTags("+", pkg.Comments)["groupName"]; override != nil {
				groupName := override[0]
				klog.V(5).Infof("overriding the group name with = %s", groupName)
				gv.Group = clientgentypes.Group(groupName)
			}
		}

		typesToRegister := []*types.Type{}
		for _, t := range pkg.Types {
			klog.V(5).Infof("considering type = %s", t.Name.String())
			for _, typeMember := range t.Members {
				if typeMember.Name == "TypeMeta" && typeMember.Embedded {
					typesToRegister = append(typesToRegister, t)
				}
			}
		}

		targets = append(targets,
			&generator.SimpleTarget{
				PkgName:       pkg.Name,
				PkgPath:       pkg.Path, // output to same pkg as input
				PkgDir:        pkg.Dir,  // output to same pkg as input
				HeaderComment: boilerplate,
				GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
					return []generator.Generator{
						&registerExternalGenerator{
							GoGenerator: generator.GoGenerator{
								OutputFilename: args.OutputFile,
							},
							gv:              gv,
							typesToGenerate: typesToRegister,
							outputPackage:   pkg.Path,
							imports:         generator.NewImportTrackerForPackage(pkg.Path),
						},
					}
				},
			})
	}

	return targets
}

// isInternal determines whether the given package
// contains the internal types or not
func isInternal(p *types.Package) (bool, error) {
	for _, t := range p.Types {
		for _, member := range t.Members {
			if member.Name == "TypeMeta" {
				return !strings.Contains(member.Tags, "json"), nil
			}
		}
	}
	return false, fmt.Errorf("unable to find TypeMeta for any types in package %s", p.Path)
}
