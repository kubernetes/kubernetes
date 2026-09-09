/*
Copyright The Kubernetes Authors.

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
	"runtime"
	"runtime/debug"

	"github.com/golangci/golangci-lint/v2/pkg/commands"
	"github.com/golangci/golangci-lint/v2/pkg/exitcodes"

	// These imports register the custom linters.
	_ "sigs.k8s.io/kube-api-linter"
	_ "sigs.k8s.io/logtools/logcheck/gclplugin"

	_ "k8s.io/kubernetes/hack/tools/golangci-lint/sorted/plugin"
)

func main() {
	// populate the golangci-lint version so the verify config command works
	golangCIVersion := ""
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, dep := range info.Deps {
			if dep.Path == "github.com/golangci/golangci-lint/v2" {
				golangCIVersion = dep.Version
				break
			}
		}
	}

	if err := commands.Execute(commands.BuildInfo{GoVersion: runtime.Version(), Version: golangCIVersion}); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "The command is terminated due to an error: %v\n", err)
		os.Exit(exitcodes.Failure)
	}
}
