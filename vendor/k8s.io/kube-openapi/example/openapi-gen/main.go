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

// This package generates openAPI definition file to be used in open API spec generation on API servers. To generate
// definition for a specific type or package add "+k8s:openapi-gen=true" tag to the type/package comment lines. To
// exclude a type from a tagged package, add "+k8s:openapi-gen=false" tag to the type comment lines.
package main

import (
	"path/filepath"

	"k8s.io/gengo/args"
	"k8s.io/kube-openapi/pkg/generators"

	"github.com/golang/glog"
)

func main() {
	arguments := args.Default()

	// Override defaults.
	arguments.OutputFileBaseName = "openapi_generated"
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