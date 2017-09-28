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
	"flag"
	"path"
	"path/filepath"

	"github.com/golang/glog"
	"github.com/spf13/pflag"

	clientgenargs "k8s.io/code-generator/cmd/client-gen/args"
	"k8s.io/code-generator/cmd/client-gen/generators"
	"k8s.io/gengo/args"
)

func main() {
	arguments := args.Default().WithoutDefaultFlagParsing()

	// Custom args.
	customArgs := &clientgenargs.CustomArgs{}
	customArgs.AddFlags(pflag.CommandLine)

	// Override defaults.
	arguments.GoHeaderFilePath = filepath.Join(args.DefaultSourceTree(), "k8s.io/kubernetes/hack/boilerplate/boilerplate.go.txt")
	arguments.CustomArgs = customArgs
	arguments.InputDirs = []string{
		"k8s.io/apimachinery/pkg/fields",
		"k8s.io/apimachinery/pkg/labels",
		"k8s.io/apimachinery/pkg/watch",
		"k8s.io/apimachinery/pkg/apimachinery/registered",
	}

	// Register default flags. We do this manually here because we have to override InputDirs below after additional
	// input dirs are parse fromt he command-line.
	arguments.AddFlags(pflag.CommandLine)
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()

	// Prefix with InputBaseDir and add client dirs as input dirs.
	for gv, pth := range customArgs.GroupVersionToInputPath {
		customArgs.GroupVersionToInputPath[gv] = path.Join(customArgs.InputBasePath, pth)
	}
	for _, pkg := range customArgs.GroupVersionToInputPath {
		arguments.InputDirs = append(arguments.InputDirs, pkg)
	}

	if err := arguments.Execute(
		generators.NameSystems(),
		generators.DefaultNameSystem(),
		generators.Packages,
	); err != nil {
		glog.Fatalf("Error: %v", err)
	}
}
