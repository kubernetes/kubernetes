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
	"io"
	"os"
	"runtime"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta3"
	pkg_runtime "github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"

	"github.com/golang/glog"
	flag "github.com/spf13/pflag"
)

var (
	functionDest = flag.StringP("func-dest", "f", "-", "Output for deep copy functions; '-' means stdout")
	version      = flag.StringP("version", "v", "v1beta3", "Version for deep copies.")
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

	knownVersion := *version
	if knownVersion == "api" {
		knownVersion = api.Scheme.Raw().InternalVersion
	}
	generator := pkg_runtime.NewDeepCopyGenerator(api.Scheme.Raw())

	for _, overwrite := range strings.Split(*overwrites, ",") {
		vals := strings.Split(overwrite, "=")
		generator.OverwritePackage(vals[0], vals[1])
	}
	for _, knownType := range api.Scheme.KnownTypes(knownVersion) {
		if err := generator.AddType(knownType); err != nil {
			glog.Errorf("error while generating deep copy functions for %v: %v", knownType, err)
		}
	}
	if err := generator.WriteImports(funcOut, *version); err != nil {
		glog.Fatalf("error while writing imports: %v", err)
	}
	if err := generator.WriteDeepCopyFunctions(funcOut); err != nil {
		glog.Fatalf("error while writing deep copy functions: %v", err)
	}
	if err := generator.RegisterDeepCopyFunctions(funcOut, *version); err != nil {
		glog.Fatalf("error while registering deep copy functions: %v", err)
	}
}
