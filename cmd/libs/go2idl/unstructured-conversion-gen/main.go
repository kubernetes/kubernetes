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

// unstructured-conversion-gen is a tool for auto-generating functions
// to convert an object to and from their unstructured representation.
//
// Generation is governed by comment tags in the source. Any package
// may request conversion generation by including a comment in the
// file-comments of one file, of the form:
//   // +k8s:unstructured-conversion-gen=true
// When generating for a package, individual types may opt out of
// conversion generation by specifying a comment on the of the form:
//   // +k8s:unstructured-conversion-gen=false
package main

import (
	"path/filepath"

	"k8s.io/gengo/args"
	"k8s.io/kubernetes/cmd/libs/go2idl/unstructured-conversion-gen/generators"

	"github.com/golang/glog"
)

func main() {
	arguments := args.Default()

	// Override default.
	arguments.OutputFileBaseName = "unstructured_conversion_generated"
	arguments.GoHeaderFilePath = filepath.Join(args.DefaultSourceTree(), "k8s.io/kubernetes/hack/boilerplate/boilerplate.go.txt")

	// Run it.
	if err := arguments.Execute(
		generators.NameSystems(),
		generators.DefaultNameSystem(),
		generators.Packages,
	); err != nil {
		glog.Fatalf("Error: %v", err)
	}
	glog.V(2).Info("Completed successfully.")
}
