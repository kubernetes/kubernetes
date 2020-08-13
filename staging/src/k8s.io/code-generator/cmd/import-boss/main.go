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

// import-boss enforces import restrictions in a given repository.
package main

import (
	"os"
	"path/filepath"

	"github.com/spf13/pflag"
	"k8s.io/code-generator/pkg/util"
	"k8s.io/gengo/args"
	"k8s.io/gengo/examples/import-boss/generators"

	"k8s.io/klog/v2"
)

func main() {
	klog.InitFlags(nil)
	arguments := args.Default()

	// Override defaults.
	arguments.GoHeaderFilePath = filepath.Join(args.DefaultSourceTree(), util.BoilerplatePath())
	pflag.CommandLine.BoolVar(&arguments.IncludeTestFiles, "include-test-files", false, "If true, include *_test.go files.")

	if err := arguments.Execute(
		generators.NameSystems(),
		generators.DefaultNameSystem(),
		generators.Packages,
	); err != nil {
		klog.Errorf("Error: %v", err)
		os.Exit(1)
	}
	klog.V(2).Info("Completed successfully.")
}
