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
	"os"

	"github.com/spf13/cobra"
)

var copyRepoFlag bool
var debugFlag bool
var onlyAdditionsFlag bool
var onlyBreakingChangesFlag bool
var quietFlag bool
var suppressReport bool
var verboseFlag bool

var rootCmd = &cobra.Command{
	Use:   "apidiff",
	Short: "Generates a diff of exported package content between two commits.",
	Long: `This tool will generate a report in JSON format describing changes to
public surface area between two specified commits.  It can work on
individual packages or a set of packages under a specified directory.`,
}

func init() {
	rootCmd.PersistentFlags().BoolVarP(&copyRepoFlag, "copyrepo", "c", false, "copy the repo instead of cloning it")
	rootCmd.PersistentFlags().BoolVarP(&debugFlag, "debug", "d", false, "debug output")
	rootCmd.PersistentFlags().BoolVarP(&onlyAdditionsFlag, "additions", "a", false, "only include additive changes in the report")
	rootCmd.PersistentFlags().BoolVarP(&onlyBreakingChangesFlag, "breakingchanges", "b", false, "only include breaking changes in the report")
	rootCmd.PersistentFlags().BoolVarP(&quietFlag, "quiet", "q", false, "suppress console output")
	rootCmd.PersistentFlags().BoolVarP(&verboseFlag, "verbose", "v", false, "verbose output")
}

// Execute executes the specified command.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(-1)
	}
}

// CommandFlags is used to specify flags when invoking commands programatically.
type CommandFlags struct {
	CopyRepo            bool
	Debug               bool
	OnlyAdditions       bool
	OnlyBreakingChanges bool
	Quiet               bool
	SuppressReport      bool
	Verbose             bool
}

// applies the specified flags to their global equivalents
func (cf CommandFlags) apply() {
	copyRepoFlag = cf.CopyRepo
	debugFlag = cf.Debug
	onlyAdditionsFlag = cf.OnlyAdditions
	onlyBreakingChangesFlag = cf.OnlyBreakingChanges
	quietFlag = cf.Quiet
	suppressReport = cf.SuppressReport
	verboseFlag = cf.Verbose
}
