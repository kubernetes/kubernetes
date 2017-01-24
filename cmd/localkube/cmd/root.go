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

package cmd

import (
	goflag "flag"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

var RootCmd = &cobra.Command{
	Use:   "localkube",
	Short: "localkube is a all-in-one kubernetes binary.",
	Long:  `localkube is a all-in-one kubernetes binary that runs all Kubernetes server binaries.`,
	Run: func(command *cobra.Command, args []string) {
		StartLocalkube()
	},
}

func init() {
	pflag.CommandLine.AddGoFlagSet(goflag.CommandLine)
}
