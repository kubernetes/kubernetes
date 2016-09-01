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

// client-gen makes the individual typed clients using go2idl.
package main

import (
	"fmt"
	"path/filepath"
	"strings"

	"k8s.io/kubernetes/cmd/libs/go2idl/args"
	clientgenargs "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/args"
	"k8s.io/kubernetes/cmd/libs/go2idl/client-gen/generators"
	"k8s.io/kubernetes/pkg/api/unversioned"

	"github.com/golang/glog"
	flag "github.com/spf13/pflag"
)

var (
	test          = flag.BoolP("test", "t", false, "set this flag to generate the client code for the testdata")
	inputVersions = flag.StringSlice("input", []string{
		"api/",
		"authentication/",
		"authorization/",
		"autoscaling/",
		"batch/",
		"certificates/",
		"extensions/",
		"rbac/",
		"storage/",
	}, "group/versions that client-gen will generate clients for. At most one version per group is allowed. Specified in the format \"group1/version1,group2/version2...\". Default to \"api/,extensions/,autoscaling/,batch/,rbac/\"")
	includedTypesOverrides = flag.StringSlice("included-types-overrides", []string{}, "list of group/version/type for which client should be generated. By default, client is generated for all types which have genclient=true in types.go. This overrides that. For each groupVersion in this list, only the types mentioned here will be included. The default check of genclient=true will be used for other group versions.")
	basePath               = flag.String("input-base", "k8s.io/kubernetes/pkg/apis", "base path to look for the api group. Default to \"k8s.io/kubernetes/pkg/apis\"")
	clientsetName          = flag.StringP("clientset-name", "n", "internalclientset", "the name of the generated clientset package.")
	clientsetPath          = flag.String("clientset-path", "k8s.io/kubernetes/pkg/client/clientset_generated/", "the generated clientset will be output to <clientset-path>/<clientset-name>. Default to \"k8s.io/kubernetes/pkg/client/clientset_generated/\"")
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
	arguments := args.Default()
	flag.Parse()
	var cmdArgs string
	flag.VisitAll(func(f *flag.Flag) {
		if !f.Changed || f.Name == "verify-only" {
			return
		}
		cmdArgs = cmdArgs + fmt.Sprintf("--%s=%s ", f.Name, f.Value)
	})

	dependencies := []string{
		"k8s.io/kubernetes/pkg/fields",
		"k8s.io/kubernetes/pkg/labels",
		"k8s.io/kubernetes/pkg/watch",
		"k8s.io/kubernetes/pkg/client/unversioned",
		"k8s.io/kubernetes/pkg/apimachinery/registered",
	}

	if *test {
		arguments.InputDirs = append(dependencies, []string{
			"k8s.io/kubernetes/cmd/libs/go2idl/client-gen/test_apis/testgroup.k8s.io",
		}...)
		arguments.CustomArgs = clientgenargs.Args{
			GroupVersions: []unversioned.GroupVersion{{Group: "testgroup.k8s.io", Version: ""}},
			GroupVersionToInputPath: map[unversioned.GroupVersion]string{
				unversioned.GroupVersion{Group: "testgroup.k8s.io", Version: ""}: "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/test_apis/testgroup.k8s.io",
			},
			ClientsetName:       "test_internalclientset",
			ClientsetOutputPath: "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/testoutput/clientset_generated/",
			ClientsetOnly:       false,
			FakeClient:          true,
			CmdArgs:             cmdArgs,
		}
	} else {
		inputPath, groupVersions, gvToPath, err := parseInputVersions()
		if err != nil {
			glog.Fatalf("Error: %v", err)
		}
		includedTypesOverrides, err := parseIncludedTypesOverrides()
		if err != nil {
			glog.Fatalf("Unexpected error: %v", err)
		}
		glog.Infof("going to generate clientset from these input paths: %v", inputPath)
		arguments.InputDirs = append(inputPath, dependencies...)

		arguments.CustomArgs = clientgenargs.Args{
			GroupVersions:           groupVersions,
			GroupVersionToInputPath: gvToPath,
			ClientsetName:           *clientsetName,
			ClientsetOutputPath:     *clientsetPath,
			ClientsetOnly:           *clientsetOnly,
			FakeClient:              *fakeClient,
			CmdArgs:                 cmdArgs,
			IncludedTypesOverrides:  includedTypesOverrides,
		}

		glog.Infof("==arguments: %v\n", arguments)
	}

	if err := arguments.Execute(
		generators.NameSystems(),
		generators.DefaultNameSystem(),
		generators.Packages,
	); err != nil {
		glog.Fatalf("Error: %v", err)
	}
}
