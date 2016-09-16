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
	"fmt"
	"os"

	"github.com/spf13/cobra/doc"
	"k8s.io/kubernetes/cmd/genutils"
	fedapiservapp "k8s.io/kubernetes/federation/cmd/federation-apiserver/app"
	fedcmapp "k8s.io/kubernetes/federation/cmd/federation-controller-manager/app"
)

// Note: We have a separate binary for generating federation docs and kube docs because of the way api groups are registered.
// If we import both kube-apiserver and federation-apiserver in the same binary then api groups from both kube and federation will get registered in both the apiservers
// and hence will produce incorrect flag values.
// We can potentially merge cmd/kubegendocs and this when we have fixed that problem.
func main() {
	// use os.Args instead of "flags" because "flags" will mess up the man pages!
	path := ""
	module := ""
	if len(os.Args) == 3 {
		path = os.Args[1]
		module = os.Args[2]
	} else {
		fmt.Fprintf(os.Stderr, "usage: %s [output directory] [module] \n", os.Args[0])
		os.Exit(1)
	}

	outDir, err := genutils.OutDir(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to get output directory: %v\n", err)
		os.Exit(1)
	}

	switch module {
	case "federation-apiserver":
		// generate docs for federated-apiserver
		apiserver := fedapiservapp.NewAPIServerCommand()
		doc.GenMarkdownTree(apiserver, outDir)
	case "federation-controller-manager":
		// generate docs for kube-controller-manager
		controllermanager := fedcmapp.NewControllerManagerCommand()
		doc.GenMarkdownTree(controllermanager, outDir)
	default:
		fmt.Fprintf(os.Stderr, "Module %s is not supported", module)
		os.Exit(1)
	}
}
