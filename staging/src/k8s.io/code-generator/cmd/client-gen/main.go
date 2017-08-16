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

// client-gen makes the individual typed clients using gengo.
package main

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	clientgenargs "k8s.io/code-generator/cmd/client-gen/args"
	"k8s.io/code-generator/cmd/client-gen/generators"
	"k8s.io/code-generator/cmd/client-gen/types"
	"k8s.io/gengo/args"

	"github.com/golang/glog"
	flag "github.com/spf13/pflag"
)

var (
	inputVersions = flag.StringSlice("input", []string{
		"api/",
		"admissionregistration/",
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
		"scheduling/",
		"settings/",
		"networking/",
	}, "group/versions that client-gen will generate clients for. At most one version per group is allowed. Specified in the format \"group1/version1,group2/version2...\".")
	includedTypesOverrides = flag.StringSlice("included-types-overrides", []string{}, "list of group/version/type for which client should be generated. By default, client is generated for all types which have genclient in types.go. This overrides that. For each groupVersion in this list, only the types mentioned here will be included. The default check of genclient will be used for other group versions.")
	basePath               = flag.String("input-base", "k8s.io/kubernetes/pkg/apis", "base path to look for the api group.")
	clientsetName          = flag.StringP("clientset-name", "n", "internalclientset", "the name of the generated clientset package.")
	clientsetAPIPath       = flag.StringP("clientset-api-path", "", "", "the value of default API path.")
	clientsetPath          = flag.String("clientset-path", "k8s.io/kubernetes/pkg/client/clientset_generated/", "the generated clientset will be output to <clientset-path>/<clientset-name>.")
	clientsetOnly          = flag.Bool("clientset-only", false, "when set, client-gen only generates the clientset shell, without generating the individual typed clients")
	fakeClient             = flag.Bool("fake-clientset", true, "when set, client-gen will generate the fake clientset that can be used in tests")
)

func versionToPath(gvPath string, group string, version string) (path string) {
	// special case for the core group
	if group == "api" {
		path = filepath.Join(*basePath, "../api", version)
	} else {
		path = filepath.Join(*basePath, gvPath, group, version)
	}
	return
}

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

func parseInputVersions() (paths []string, groups []types.GroupVersions, gvToPath map[types.GroupVersion]string, err error) {
	var seenGroups = make(map[types.Group]*types.GroupVersions)
	gvToPath = make(map[types.GroupVersion]string)
	for _, input := range *inputVersions {
		gvPath, gvString := parsePathGroupVersion(input)
		gv, err := types.ToGroupVersion(gvString)
		if err != nil {
			return nil, nil, nil, err
		}
		if group, ok := seenGroups[gv.Group]; ok {
			(*seenGroups[gv.Group]).Versions = append(group.Versions, gv.Version)
		} else {
			seenGroups[gv.Group] = &types.GroupVersions{
				Group:    gv.Group,
				Versions: []types.Version{gv.Version},
			}
		}

		path := versionToPath(gvPath, gv.Group.String(), gv.Version.String())
		paths = append(paths, path)
		gvToPath[gv] = path
	}
	var groupNames []string
	for groupName := range seenGroups {
		groupNames = append(groupNames, groupName.String())
	}
	sort.Strings(groupNames)
	for _, groupName := range groupNames {
		groups = append(groups, *seenGroups[types.Group(groupName)])
	}

	return paths, groups, gvToPath, nil
}

func parseIncludedTypesOverrides() (map[types.GroupVersion][]string, error) {
	overrides := make(map[types.GroupVersion][]string)
	for _, input := range *includedTypesOverrides {
		gvString, typeStr, err := parseGroupVersionType(input)
		if err != nil {
			return nil, err
		}
		gv, err := types.ToGroupVersion(gvString)
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
	arguments := args.Default()
	arguments.GoHeaderFilePath = filepath.Join(args.DefaultSourceTree(), "k8s.io/kubernetes/hack/boilerplate/boilerplate.go.txt")
	flag.Parse()
	var cmdArgs string
	flag.VisitAll(func(f *flag.Flag) {
		if !f.Changed || f.Name == "verify-only" {
			return
		}
		cmdArgs = cmdArgs + fmt.Sprintf("--%s=%s ", f.Name, f.Value)
	})

	dependencies := []string{
		"k8s.io/apimachinery/pkg/fields",
		"k8s.io/apimachinery/pkg/labels",
		"k8s.io/apimachinery/pkg/watch",
		"k8s.io/apimachinery/pkg/apimachinery/registered",
	}

	inputPath, groups, gvToPath, err := parseInputVersions()
	if err != nil {
		glog.Fatalf("Error: %v", err)
	}
	includedTypesOverrides, err := parseIncludedTypesOverrides()
	if err != nil {
		glog.Fatalf("Unexpected error: %v", err)
	}
	glog.V(3).Infof("going to generate clientset from these input paths: %v", inputPath)
	arguments.InputDirs = append(inputPath, dependencies...)

	arguments.CustomArgs = clientgenargs.Args{
		Groups:                  groups,
		GroupVersionToInputPath: gvToPath,
		ClientsetName:           *clientsetName,
		ClientsetAPIPath:        *clientsetAPIPath,
		ClientsetOutputPath:     *clientsetPath,
		ClientsetOnly:           *clientsetOnly,
		FakeClient:              *fakeClient,
		CmdArgs:                 cmdArgs,
		IncludedTypesOverrides:  includedTypesOverrides,
	}

	glog.V(3).Infof("==arguments: %v\n", arguments)

	if err := arguments.Execute(
		generators.NameSystems(),
		generators.DefaultNameSystem(),
		generators.Packages,
	); err != nil {
		glog.Fatalf("Error: %v", err)
	}
}
