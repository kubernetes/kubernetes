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
	"flag"
	"path/filepath"

	"github.com/golang/glog"
	"github.com/spf13/pflag"

	clientgenargs "k8s.io/code-generator/cmd/client-gen/args"
	"k8s.io/code-generator/cmd/lister-gen/generators"
	"k8s.io/gengo/args"
)

func main() {
	arguments := args.Default().WithoutDefaultFlagParsing()

	namerExceptionFile := pflag.StringP("namer-exceptions-file", "", "", "a YAML file that contains namer exceptions.")

	// Override defaults.
	arguments.GoHeaderFilePath = filepath.Join(args.DefaultSourceTree(), "k8s.io/kubernetes/hack/boilerplate/boilerplate.go.txt")
	arguments.OutputPackagePath = "k8s.io/kubernetes/pkg/client/listers"

	arguments.AddFlags(pflag.CommandLine)
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()

	exceptions, err := clientgenargs.LoadNamerExceptions(*namerExceptionFile)
	if err != nil {
		glog.Fatalf("Unable to load namer exceptions from %s: %v", namerExceptionFile, err)
	}

	// Run it.
	if err := arguments.Execute(
		generators.NameSystems(exceptions),
		generators.DefaultNameSystem(),
		generators.Packages,
	); err != nil {
		glog.Fatalf("Error: %v", err)
	}
	glog.V(2).Info("Completed successfully.")
}
