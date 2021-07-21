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

// set-gen is an example usage of gengo.
//
// Structs in the input directories with the below line in their comments will
// have sets generated for them.
// // +genset
//
// Any builtin type referenced anywhere in the input directories will have a
// set generated for it.
package main

import (
	"os"

	"k8s.io/code-generator/pkg/util"
	"k8s.io/gengo/args"
	"k8s.io/gengo/examples/set-gen/generators"

	"k8s.io/klog/v2"
)

func main() {
	klog.InitFlags(nil)
	arguments := args.Default()

	// Override defaults.
	arguments.GoHeaderFilePath = util.BoilerplatePath()
	arguments.InputDirs = []string{"k8s.io/kubernetes/pkg/util/sets/types"}
	arguments.OutputPackagePath = "k8s.io/apimachinery/pkg/util/sets"

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
