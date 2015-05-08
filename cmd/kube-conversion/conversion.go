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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta3"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"

	"github.com/golang/glog"
	flag "github.com/spf13/pflag"
)

var (
	functionDest = flag.StringP("funcDest", "f", "-", "Output for conversion functions; '-' means stdout")
	namesDest    = flag.StringP("nameDest", "n", "-", "Output for function names; '-' means stdout")
	version      = flag.StringP("version", "v", "v1beta3", "Version for conversion.")
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
	var nameOut io.Writer
	if *namesDest == "-" {
		nameOut = os.Stdout
	} else {
		file, err := os.Create(*namesDest)
		if err != nil {
			glog.Fatalf("Couldn't open %v: %v", *functionDest, err)
		}
		defer file.Close()
		nameOut = file
	}

	generator := conversion.NewGenerator(api.Scheme.Raw())
	// TODO(wojtek-t): Change the overwrites to a flag.
	generator.OverwritePackage(*version, "")
	generator.OverwritePackage("api", "newer")
	for _, knownType := range api.Scheme.KnownTypes(*version) {
		if err := generator.GenerateConversionsForType(*version, knownType); err != nil {
			glog.Errorf("error while generating conversion functions for %v: %v", knownType, err)
		}
	}
	if err := generator.WriteConversionFunctions(funcOut); err != nil {
		glog.Fatalf("Error while writing conversion functions: %v", err)
	}
	if err := generator.WriteConversionFunctionNames(nameOut); err != nil {
		glog.Fatalf("Error while writing conversion functions: %v", err)
	}
}
