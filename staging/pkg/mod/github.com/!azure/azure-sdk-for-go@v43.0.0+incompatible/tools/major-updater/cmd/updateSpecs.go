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
	"path/filepath"

	"github.com/Azure/azure-sdk-for-go/tools/apidiff/repo"
	"github.com/spf13/cobra"
)

var updateSpecsCmd = &cobra.Command{
	Use:   "updateSpec <spec dir>",
	Short: "Update the specs repo on master branch",
	Long: `This command will change the working directory to the specs folder,
	checkout to master branch and update it`,
	Args: func(cmd *cobra.Command, args []string) error {
		return cobra.ExactArgs(1)(cmd, args)
	},
	RunE: func(cmd *cobra.Command, args []string) error {
		specs := args[0]
		err := theUpdateSpecsCommand(specs)
		return err
	},
}

func init() {
	rootCmd.AddCommand(updateSpecsCmd)
}

func theUpdateSpecsCommand(spec string) error {
	vprintln("Updating specs repo...")
	absolutePathOfSpec, err := filepath.Abs(spec)
	if err != nil {
		return fmt.Errorf("failed to get the directory of specs: %v", err)
	}
	err = changeDir(absolutePathOfSpec)
	if err != nil {
		return fmt.Errorf("failed to change directory to %s: %v", absolutePathOfSpec, err)
	}
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get the current working directory: %v", err)
	}
	wt, err := repo.Get(cwd)
	if err != nil {
		return fmt.Errorf("failed to get the working tree: %v", err)
	}
	vprintf("Checking out to %s branch in %s\n", master, cwd)
	err = wt.Checkout(master)
	if err != nil {
		return fmt.Errorf("checkout failed: %v", err)
	}
	vprintf("Pulling %s branch in %s\n", master, cwd)
	err = wt.Pull(specUpstream, master)
	if err != nil {
		return fmt.Errorf("pull failed: %v", err)
	}
	return nil
}
