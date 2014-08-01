/*
Copyright 2014 Google Inc. All rights reserved.

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

// Package flag defines utility functions to handle command line flags related to version of Kubernetes.
package flag

import (
	"flag"
	"fmt"
	"os"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
)

var (
	versionFlag = flag.Bool("version", false, "Print version information and quit")
)

// PrintAndExitIfRequested will check if the -version flag was passed
// and, if so, print the version and exit.
func PrintAndExitIfRequested() {
	if *versionFlag {
		fmt.Printf("Kubernetes %s\n", version.Get())
		os.Exit(0)
	}
}
