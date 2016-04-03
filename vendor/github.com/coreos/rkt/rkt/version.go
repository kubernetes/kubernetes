// Copyright 2014-2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"runtime"

	"github.com/appc/spec/schema"
	"github.com/coreos/rkt/version"
	"github.com/spf13/cobra"
)

var (
	rktFeatures string // set by linker
	cmdVersion  = &cobra.Command{
		Use:   "version",
		Short: "Print the version and exit",
		Long:  "Print the version of rkt, and various build and configuration information.",
		Run:   runWrapper(runVersion),
	}
)

func init() {
	cmdRkt.AddCommand(cmdVersion)
}

func runVersion(cmd *cobra.Command, args []string) (exit int) {
	stdout.Printf("rkt Version: %s", version.Version)
	stdout.Printf("appc Version: %s", schema.AppContainerVersion)
	stdout.Printf("Go Version: %s", runtime.Version())
	stdout.Printf("Go OS/Arch: %s/%s", runtime.GOOS, runtime.GOARCH)
	stdout.Printf("Features: %s", rktFeatures)
	return
}
