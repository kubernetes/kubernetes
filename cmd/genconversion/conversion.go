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

	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/install"
	"k8s.io/kubernetes/pkg/api/unversioned"
	_ "k8s.io/kubernetes/pkg/apis/autoscaling/install"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	_ "k8s.io/kubernetes/pkg/apis/componentconfig/install"
	_ "k8s.io/kubernetes/pkg/apis/extensions/install"
	_ "k8s.io/kubernetes/pkg/apis/metrics/install"
	kruntime "k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/golang/glog"
	flag "github.com/spf13/pflag"
	"golang.org/x/tools/imports"
)

const pkgBase = "k8s.io/kubernetes/pkg"

var (
	functionDest = flag.StringP("funcDest", "f", "-", "Output for conversion functions; '-' means stdout")
	groupVersion = flag.StringP("version", "v", "api/v1", "groupPath/version for conversion.")
)

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

	gv, err := unversioned.ParseGroupVersion(*groupVersion)
	if err != nil {
		glog.Fatalf("Error parsing groupversion %v: %v", *groupVersion, err)
	}

	_, err = data.WriteString(fmt.Sprintf("package %v\n", gv.Version))
	if err != nil {
		glog.Fatalf("Error while writing package line: %v", err)
	}

	versionPath := pkgPath(gv.Group, gv.Version)
	generator := kruntime.NewConversionGenerator(api.Scheme, versionPath)
	apiShort := generator.AddImport(path.Join(pkgBase, "api"))
	generator.AddImport(path.Join(pkgBase, "api/resource"))
	// TODO(wojtek-t): Change the overwrites to a flag.
	generator.OverwritePackage(gv.Version, "")
	for _, knownType := range api.Scheme.KnownTypes(gv) {
		if knownType.PkgPath() != versionPath {
			continue
		}
		if err := generator.GenerateConversionsForType(gv, knownType); err != nil {
			glog.Errorf("Error while generating conversion functions for %v: %v", knownType, err)
		}
	}
	generator.RepackImports(sets.NewString())
	if err := generator.WriteImports(data); err != nil {
		glog.Fatalf("Error while writing imports: %v", err)
	}
	if err := generator.WriteConversionFunctions(data); err != nil {
		glog.Fatalf("Error while writing conversion functions: %v", err)
	}
	if err := generator.RegisterConversionFunctions(data, fmt.Sprintf("%s.Scheme", apiShort)); err != nil {
		glog.Fatalf("Error while writing conversion functions: %v", err)
	}

	b, err := imports.Process("", data.Bytes(), nil)
	if err != nil {
		for i, s := range bytes.Split(data.Bytes(), []byte("\n")) {
			glog.Infof("%d:\t%s", i, s)
		}
		glog.Fatalf("Error while update imports: %v\n", err)
	}
	if _, err := funcOut.Write(b); err != nil {
		glog.Fatalf("Error while writing out the resulting file: %v", err)
	}
}
