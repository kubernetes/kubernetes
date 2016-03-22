/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

// conversion-gen is a tool for auto-generating Conversion functions.
//
// Structs in the input directories with the below line in their comments
// will be ignored during generation.
// // +genconversion=false
package main

import (
	"k8s.io/kubernetes/cmd/libs/go2idl/args"
	"k8s.io/kubernetes/cmd/libs/go2idl/conversion-gen/generators"

	"github.com/golang/glog"
)

func main() {
	arguments := args.Default()

	// Override defaults. These are Kubernetes specific input locations.
	arguments.InputDirs = []string{
		"k8s.io/kubernetes/pkg/api/v1",
	}

	if err := arguments.Execute(
		generators.NameSystems(),
		generators.DefaultNameSystem(),
		generators.Packages,
	); err != nil {
		glog.Fatalf("Error: %v", err)
	}
	glog.Info("Completed successfully.")
}
