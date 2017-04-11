/*
Copyright 2017 The Kubernetes Authors.

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
	"os"
	"runtime"

	"github.com/golang/glog"
	"k8s.io/apiserver-builder/cmd/apiregister-gen/generators"
	"k8s.io/apiserver/pkg/util/logs"
	"k8s.io/gengo/args"
)

func main() {
	logs.InitLogs()
	defer logs.FlushLogs()

	if len(os.Getenv("GOMAXPROCS")) == 0 {
		runtime.GOMAXPROCS(runtime.NumCPU())
	}

	arguments := args.Default()

	// Override defaults.
	arguments.OutputFileBaseName = "zz_generated.api.register"

	// Custom args.
	customArgs := &generators.CustomArgs{}
	arguments.CustomArgs = customArgs

	g := generators.Gen{}
	if err := g.Execute(arguments); err != nil {
		glog.Fatalf("Error: %v", err)
	}
	glog.V(2).Info("Completed successfully.")
}
