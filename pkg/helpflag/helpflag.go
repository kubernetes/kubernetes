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

// Package helpflag defines general help flag for cmds such as kubelet, kube-proxy, etc
package helpflag

import (
	"fmt"
	"os"

	"github.com/spf13/pflag"
)

var (
	help = pflag.BoolP("help", "h", false, "display this help and exit")
)

// PrintAndExitIfRequested prints help information and then exits zero value if -h or --help flag is detected
func PrintAndExitIfRequested() {
	if *help {
		fs := pflag.CommandLine
		fs.SetOutput(os.Stdout)
		fmt.Fprintf(os.Stdout, "Usage of %s:\n", os.Args[0])
		fs.PrintDefaults()
		os.Exit(0)
	}
}
