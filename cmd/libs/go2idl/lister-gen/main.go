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

package main

import (
	"fmt"
	"path/filepath"
	"strings"

	"k8s.io/gengo/args"
	listergenargs "k8s.io/kubernetes/cmd/libs/go2idl/lister-gen/args"
	"k8s.io/kubernetes/cmd/libs/go2idl/lister-gen/generators"
	"k8s.io/kubernetes/pkg/api/unversioned"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

var (
	// TODO make these common between client-gen and lister-gen, e.g. by using
	// cobra
	inputVersions = pflag.StringSlice("input", []string{
		"api/",
		"authentication/",
		"authorization/",
		"autoscaling/",
		"batch/",
		"certificates/",
		"extensions/",
		"rbac/",
		"storage/",
		"apps/",
		"policy/",
	}, "group/versions that client-gen will generate clients for. At most one version per group is allowed. Specified in the format \"group1/version1,group2/version2...\".")
	includedTypesOverrides = pflag.StringSlice("included-types-overrides", []string{}, "list of group/version/type for which client should be generated. By default, client is generated for all types which have genclient=true in types.go. This overrides that. For each groupVersion in this list, only the types mentioned here will be included. The default check of genclient=true will be used for other group versions.")
	basePath               = pflag.String("input-base", "k8s.io/kubernetes/pkg/apis", "base path to look for the api group.")
)

// TODO move this to a common, exported function so client-gen and lister-gen
// can both use it.
func versionToPath(gvPath string, group string, version string) (path string) {
	// special case for the core group
	if group == "api" {
		path = filepath.Join(*basePath, "../api", version)
	} else {
		path = filepath.Join(*basePath, gvPath, group, version)
	}
	return
}

// TODO move this to a common, exported function so client-gen and lister-gen
// can both use it.
func parseGroupVersionType(gvtString string) (gvString string, typeStr string, err error) {
	invalidFormatErr := fmt.Errorf("invalid value: %s, should be of the form group/version/type", gvtString)
	subs := strings.Split(gvtString, "/")
	length := len(subs)
	switch length {
	case 2:
		// gvtString of the form group/type, e.g. api/Service,extensions/ReplicaSet
		return subs[0] + "/", subs[1], nil
	case 3:
		return strings.Join(subs[:length-1], "/"), subs[length-1], nil
	default:
		return "", "", invalidFormatErr
	}
}

// TODO move this to a common, exported function so client-gen and lister-gen
// can both use it.
func parsePathGroupVersion(pgvString string) (gvPath string, gvString string) {
	subs := strings.Split(pgvString, "/")
	length := len(subs)
	switch length {
	case 0, 1, 2:
		return "", pgvString
	default:
		return strings.Join(subs[:length-2], "/"), strings.Join(subs[length-2:], "/")
	}
}

// TODO move this to a common, exported function so client-gen and lister-gen
// can both use it.
func parseInputVersions() (paths []string, groupVersions []unversioned.GroupVersion, gvToPath map[unversioned.GroupVersion]string, err error) {
	var visitedGroups = make(map[string]struct{})
	gvToPath = make(map[unversioned.GroupVersion]string)
	for _, input := range *inputVersions {
		gvPath, gvString := parsePathGroupVersion(input)
		gv, err := unversioned.ParseGroupVersion(gvString)
		if err != nil {
			return nil, nil, nil, err
		}

		if _, found := visitedGroups[gv.Group]; found {
			return nil, nil, nil, fmt.Errorf("group %q appeared more than once in the input. At most one version is allowed for each group.", gv.Group)
		}
		visitedGroups[gv.Group] = struct{}{}
		groupVersions = append(groupVersions, gv)
		path := versionToPath(gvPath, gv.Group, gv.Version)
		paths = append(paths, path)
		gvToPath[gv] = path
	}
	return paths, groupVersions, gvToPath, nil
}

// TODO move this to a common, exported function so client-gen and lister-gen
// can both use it.
func parseIncludedTypesOverrides() (map[unversioned.GroupVersion][]string, error) {
	overrides := make(map[unversioned.GroupVersion][]string)
	for _, input := range *includedTypesOverrides {
		gvString, typeStr, err := parseGroupVersionType(input)
		if err != nil {
			return nil, err
		}
		gv, err := unversioned.ParseGroupVersion(gvString)
		if err != nil {
			return nil, err
		}
		types, ok := overrides[gv]
		if !ok {
			types = []string{}
		}
		types = append(types, typeStr)
		overrides[gv] = types
	}
	return overrides, nil
}

func main() {
	arguments := &args.GeneratorArgs{
		OutputBase:         args.DefaultSourceTree(),
		GoHeaderFilePath:   filepath.Join(args.DefaultSourceTree(), "k8s.io/kubernetes/hack/boilerplate/boilerplate.go.txt"),
		GeneratedBuildTag:  "ignore_autogenerated",
		OutputFileBaseName: "zz_generated.listers",
		OutputPackagePath:  "k8s.io/kubernetes/pkg/client/lister",
	}
	arguments.AddFlags(pflag.CommandLine)

	dependencies := []string{
		"k8s.io/kubernetes/pkg/fields",
		"k8s.io/kubernetes/pkg/labels",
		"k8s.io/kubernetes/pkg/watch",
		"k8s.io/kubernetes/pkg/client/unversioned",
		"k8s.io/kubernetes/pkg/apimachinery/registered",
	}

	inputPath, groupVersions, gvToPath, err := parseInputVersions()
	if err != nil {
		glog.Fatalf("Error: %v", err)
	}
	includedTypesOverrides, err := parseIncludedTypesOverrides()
	if err != nil {
		glog.Fatalf("Unexpected error: %v", err)
	}
	glog.V(3).Infof("going to generate listers from these input paths: %v", inputPath)
	arguments.InputDirs = append(inputPath, dependencies...)

	arguments.CustomArgs = listergenargs.Args{
		GroupVersions:           groupVersions,
		GroupVersionToInputPath: gvToPath,
		IncludedTypesOverrides:  includedTypesOverrides,
	}

	glog.V(3).Infof("==arguments: %v\n", arguments)

	// Run it.
	if err := arguments.Execute(
		generators.NameSystems(),
		generators.DefaultNameSystem(),
		generators.Packages,
	); err != nil {
		glog.Fatalf("Error: %v", err)
	}
	glog.V(2).Info("Completed successfully.")
}
