/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/cmd/libs/go2idl/args"
	clientgenargs "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/args"
	"k8s.io/kubernetes/cmd/libs/go2idl/client-gen/generators"
	"k8s.io/kubernetes/pkg/api/unversioned"

	"github.com/golang/glog"
	flag "github.com/spf13/pflag"
)

var (
	test          = flag.BoolP("test", "t", false, "set this flag to generate the client code for the testdata")
	inputVersions = flag.StringSlice("input", []string{"api/", "extensions/"}, "group/versions that client-gen will generate clients for. At most one version per group is allowed. Specified in the format \"group1/version1,group2/version2...\". Default to \"api/,extensions\"")
	clientsetName = flag.StringP("clientset-name", "n", "internalclientset", "the name of the generated clientset package.")
	clientsetPath = flag.String("clientset-path", "k8s.io/kubernetes/pkg/client/clientset_generated/", "the generated clientset will be output to <clientset-path>/<clientset-name>. Default to \"k8s.io/kubernetes/pkg/client/clientset_generated/\"")
	clientsetOnly = flag.Bool("clientset-only", false, "when set, client-gen only generates the clientset shell, without generating the individual typed clients")
	fakeClient    = flag.Bool("fake-clientset", true, "when set, client-gen will generate the fake clientset that can be used in tests")
)

func versionToPath(group string, version string) (path string) {
	const base = "k8s.io/kubernetes/pkg"
	// special case for the core group
	if group == "api" {
		path = filepath.Join(base, "api", version)
	} else {
		path = filepath.Join(base, "apis", group, version)
	}
	return
}

func parseInputVersions() (paths []string, groupVersions []unversioned.GroupVersion, gvToPath map[unversioned.GroupVersion]string, err error) {
	var visitedGroups = make(map[string]struct{})
	gvToPath = make(map[unversioned.GroupVersion]string)
	for _, gvString := range *inputVersions {
		gv, err := unversioned.ParseGroupVersion(gvString)
		if err != nil {
			return nil, nil, nil, err
		}

		if _, found := visitedGroups[gv.Group]; found {
			return nil, nil, nil, fmt.Errorf("group %q appeared more than once in the input. At most one version is allowed for each group.", gv.Group)
		}
		visitedGroups[gv.Group] = struct{}{}
		groupVersions = append(groupVersions, gv)
		path := versionToPath(gv.Group, gv.Version)
		paths = append(paths, path)
		gvToPath[gv] = path
	}
	return paths, groupVersions, gvToPath, nil
}

func main() {
	arguments := args.Default()
	flag.Parse()
	dependencies := []string{
		"k8s.io/kubernetes/pkg/fields",
		"k8s.io/kubernetes/pkg/labels",
		"k8s.io/kubernetes/pkg/watch",
		"k8s.io/kubernetes/pkg/client/unversioned",
		"k8s.io/kubernetes/pkg/apimachinery/registered",
	}

	if *test {
		arguments.InputDirs = append(dependencies, []string{
			"k8s.io/kubernetes/cmd/libs/go2idl/client-gen/testdata/apis/testgroup",
		}...)
		// We may change the output path later.
		arguments.OutputPackagePath = "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/testoutput"
		arguments.CustomArgs = clientgenargs.Args{
			[]unversioned.GroupVersion{{"testgroup", ""}},
			map[unversioned.GroupVersion]string{
				unversioned.GroupVersion{"testgroup", ""}: "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/testdata/apis/testgroup",
			},
			"test_internalclientset",
			"k8s.io/kubernetes/cmd/libs/go2idl/client-gen/testoutput/clientset_generated/",
			false,
			false,
		}
	} else {
		inputPath, groupVersions, gvToPath, err := parseInputVersions()
		if err != nil {
			glog.Fatalf("Error: %v", err)
		}
		glog.Infof("going to generate clientset from these input paths: %v", inputPath)
		arguments.InputDirs = append(inputPath, dependencies...)
		// TODO: we need to make OutPackagePath a map[string]string. For example,
		// we need clientset and the individual typed clients be output to different
		// output path.

		// We may change the output path later.
		arguments.OutputPackagePath = "k8s.io/kubernetes/pkg/client/typed/generated"

		arguments.CustomArgs = clientgenargs.Args{
			groupVersions,
			gvToPath,
			*clientsetName,
			*clientsetPath,
			*clientsetOnly,
			*fakeClient,
		}
	}

	if err := arguments.Execute(
		generators.NameSystems(),
		generators.DefaultNameSystem(),
		generators.Packages,
	); err != nil {
		glog.Fatalf("Error: %v", err)
	}
}
