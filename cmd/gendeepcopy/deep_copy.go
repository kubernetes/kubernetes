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
	apiutil "k8s.io/kubernetes/pkg/api/util"
	_ "k8s.io/kubernetes/pkg/api/v1"
	_ "k8s.io/kubernetes/pkg/expapi"
	_ "k8s.io/kubernetes/pkg/expapi/v1alpha1"
	pkg_runtime "k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"

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

	group := apiutil.GetGroup(*groupVersion)
	version := apiutil.GetVersion(*groupVersion)
	registerTo := "api.Scheme"
	if *groupVersion == "api/" {
		registerTo = "Scheme"
	}
	pkgname := group
	if group == "experimental" {
		pkgname = "expapi"
	}
	if len(version) != 0 {
		pkgname = version
	}

	_, err := data.WriteString(fmt.Sprintf("package %s\n", pkgname))
	if err != nil {
		glog.Fatalf("error writing package line: %v", err)
	}

	versionPath := path.Join(pkgBase, *groupVersion)
	if group == "experimental" {
		//TODO: we should rename the direcotry /expapi to /experimental, so directory path match groupVerson
		versionPath = path.Join(pkgBase, "expapi", version)
	}
	generator := pkg_runtime.NewDeepCopyGenerator(api.Scheme.Raw(), versionPath, util.NewStringSet("k8s.io/kubernetes"))
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
	//handle the special cases of groupVersion = "api/" and "experimental/"
	if version == "" {
		*groupVersion = ""
	}
	if *groupVersion == "api/v1" {
		*groupVersion = "v1"
	}
	for _, knownType := range api.Scheme.KnownTypes(*groupVersion) {
		if !strings.HasPrefix(knownType.PkgPath(), versionPath) {
			continue
		}
		if err := generator.AddType(knownType); err != nil {
			glog.Errorf("error while generating deep copy functions for %v: %v", knownType, err)
		}
	}
	generator.RepackImports()
	if err := generator.WriteImports(data); err != nil {
		glog.Fatalf("error while writing imports: %v", err)
	}
	if err := generator.WriteDeepCopyFunctions(data); err != nil {
		glog.Fatalf("error while writing deep copy functions: %v", err)
	}
	if err := generator.RegisterDeepCopyFunctions(data, registerTo); err != nil {
		glog.Fatalf("error while registering deep copy functions: %v", err)
	}
	b, err := imports.Process("", data.Bytes(), nil)
	if err != nil {
		glog.Fatalf("error while update imports: %v", err)
	}
	if _, err := funcOut.Write(b); err != nil {
		glog.Fatalf("error while writing out the resulting file: %v", err)
	}
}
