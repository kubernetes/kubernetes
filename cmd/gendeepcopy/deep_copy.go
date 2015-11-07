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

package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path"
	"runtime"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/v1"
	_ "k8s.io/kubernetes/pkg/apis/componentconfig"
	_ "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	_ "k8s.io/kubernetes/pkg/apis/extensions"
	_ "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	_ "k8s.io/kubernetes/pkg/apis/metrics"
	_ "k8s.io/kubernetes/pkg/apis/metrics/v1alpha1"
	kruntime "k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/golang/glog"
	flag "github.com/spf13/pflag"
	"golang.org/x/tools/imports"
)

const pkgBase = "k8s.io/kubernetes/pkg"

var (
	functionDest = flag.StringP("funcDest", "f", "-", "Output for deep copy functions; '-' means stdout")
	groupVersion = flag.StringP("version", "v", "", "groupPath/version for deep copies.")
	overwrites   = flag.StringP("overwrites", "o", "", "Comma-separated overwrites for package names")
)

// types inside the api package don't need to say "api.Scheme"; all others do.
func destScheme(group, version string) string {
	if group == "" && version == "" {
		return "Scheme"
	}
	return "api.Scheme"
}

// We're moving to pkg/apis/group/version. This handles new and legacy packages.
func pkgPath(group, version string) string {
	if group == "" {
		group = "api"
	}
	gv := group
	if version != "" {
		gv = path.Join(group, version)
	}
	switch {
	case group == "api":
		// TODO(lavalamp): remove this special case when we move api to apis/api
		return path.Join(pkgBase, gv)
	default:
		return path.Join(pkgBase, "apis", gv)
	}
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	flag.Parse()

	var funcOut io.Writer
	if *functionDest == "-" {
		funcOut = os.Stdout
	} else {
		file, err := os.Create(*functionDest)
		if err != nil {
			glog.Fatalf("Couldn't open %v: %v", *functionDest, err)
		}
		defer file.Close()
		funcOut = file
	}

	data := new(bytes.Buffer)

	group, version := path.Split(*groupVersion)
	group = strings.TrimRight(group, "/")
	registerTo := destScheme(group, version)
	var pkgname string
	if group == "" {
		// the internal version of v1 is registered in package api
		pkgname = "api"
	} else {
		pkgname = group
	}
	if len(version) != 0 {
		pkgname = version
	}

	_, err := data.WriteString(fmt.Sprintf("package %s\n", pkgname))
	if err != nil {
		glog.Fatalf("Error while writing package line: %v", err)
	}

	versionPath := pkgPath(group, version)
	generator := kruntime.NewDeepCopyGenerator(api.Scheme.Raw(), versionPath, sets.NewString("k8s.io/kubernetes"))
	generator.AddImport(path.Join(pkgBase, "api"))

	if len(*overwrites) > 0 {
		for _, overwrite := range strings.Split(*overwrites, ",") {
			if !strings.Contains(overwrite, "=") {
				glog.Fatalf("Invalid overwrite syntax: %s", overwrite)
			}
			vals := strings.Split(overwrite, "=")
			generator.OverwritePackage(vals[0], vals[1])
		}
	}
	var schemeVersion string
	if version == "" {
		// This occurs when we generate deep-copy for internal version.
		schemeVersion = ""
	} else {
		schemeVersion = *groupVersion
	}
	for _, knownType := range api.Scheme.KnownTypes(schemeVersion) {
		if knownType.PkgPath() != versionPath {
			continue
		}
		if err := generator.AddType(knownType); err != nil {
			glog.Errorf("Error while generating deep copy functions for %v: %v", knownType, err)
		}
	}
	generator.RepackImports()
	if err := generator.WriteImports(data); err != nil {
		glog.Fatalf("Error while writing imports: %v", err)
	}
	if err := generator.WriteDeepCopyFunctions(data); err != nil {
		glog.Fatalf("Error while writing deep copy functions: %v", err)
	}
	if err := generator.RegisterDeepCopyFunctions(data, registerTo); err != nil {
		glog.Fatalf("Error while registering deep copy functions: %v", err)
	}
	b, err := imports.Process("", data.Bytes(), nil)
	if err != nil {
		glog.Fatalf("Error while update imports: %v", err)
	}
	if _, err := funcOut.Write(b); err != nil {
		glog.Fatalf("Error while writing out the resulting file: %v", err)
	}
}
