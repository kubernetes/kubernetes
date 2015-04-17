/*
Copyright 2015 Google Inc. All rights reserved.

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
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta3"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"

	"github.com/golang/glog"
	flag "github.com/spf13/pflag"
)

var (
	outputDest = flag.StringP("output", "o", "-", "Output destination; '-' means stdout")
	versions   = flag.StringP("versions", "v", "v1beta3", "Comma separated list of versions for conversion.")
)

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	flag.Parse()

	var out io.Writer
	if *outputDest == "-" {
		out = os.Stdout
	} else {
		file, err := os.Create(*outputDest)
		if err != nil {
			glog.Fatalf("Couldn't open %v: %v", *outputDest, err)
		}
		defer file.Close()
		out = file
	}

	versionsForConversion := strings.Split(*versions, ",")
	for _, version := range versionsForConversion {
		generator := conversion.NewGenerator(api.Scheme.Raw())
		// TODO(wojtek-t): Change the overwrites to a flag.
		generator.OverwritePackage(version, "")
		generator.OverwritePackage("api", "newer")
		for _, knownType := range api.Scheme.KnownTypes(version) {
			if err := generator.GenerateConversionsForType(version, knownType); err != nil {
				glog.Errorf("error while generating conversion functions for %v: %v", knownType, err)
			}
		}
		if err := generator.WriteConversionFunctions(out); err != nil {
			glog.Fatalf("Error while writing conversion functions: %v", err)
		}
	}
}
