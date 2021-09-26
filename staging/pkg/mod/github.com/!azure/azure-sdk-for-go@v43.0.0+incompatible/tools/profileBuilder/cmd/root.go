// +build go1.9

// Copyright 2018 Microsoft Corporation
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

package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

// command line options common to all commands
var (
	clearOutputFlag bool
	verboseFlag     bool
	modulesFlag     bool
	profileName     string
	outputRootDir   string
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "profileBuilder",
	Short: "Creates virtualized packages to simplify multi-API Version applications.",
	Long: `A profile is a virtualized set of packages, which attempts to hide the
complexity of choosing API Versions from customers who don't need the
flexiblity of separating the version of the Azure SDK for Go they're employing
from the version of Azure services they are targeting.

"profileBuilder" does the heavy-lifting of creating those virtualized packages.
Each of the sub-commands of profileBuilder applies a different strategy for
choosing which packages to include in the profile.
`,
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func init() {
	rootCmd.PersistentFlags().BoolVarP(&clearOutputFlag, "clear-output", "c", false, "Removes any directories in the output-folder before writing a profile.")
	rootCmd.PersistentFlags().BoolVarP(&verboseFlag, "verbose", "v", false, "Use stderr to log verbose output.")
	rootCmd.PersistentFlags().BoolVarP(&modulesFlag, "modules", "m", false, "Executes commands in modules-aware mode.")
	rootCmd.PersistentFlags().StringVarP(&profileName, "name", "n", "", "The name that should be used to identify the profile.")
	rootCmd.PersistentFlags().StringVarP(&outputRootDir, "output-location", "o", "", "The folder in which to output the generated profile.")
	rootCmd.MarkPersistentFlagRequired("name")
	rootCmd.MarkPersistentFlagRequired("output-location")
}
