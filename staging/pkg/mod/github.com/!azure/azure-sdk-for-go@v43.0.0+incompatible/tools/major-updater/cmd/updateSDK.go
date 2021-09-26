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
	"strconv"

	"github.com/Azure/azure-sdk-for-go/tools/apidiff/repo"
	"github.com/spf13/cobra"
)

var updateSDKCmd = &cobra.Command{
	Use:   "updateSDK <SDK dir>",
	Short: "Update the SDK repo on latest branch",
	Long: `This command will checkout to latest branch in SDK repo, 
	find next major version number based on tags, then create a new branch based on the latest branch`,
	Args: func(cmd *cobra.Command, args []string) error {
		return cobra.ExactArgs(1)(cmd, args)
	},
	RunE: func(cmd *cobra.Command, args []string) error {
		sdk := args[0]
		err := theUpdateSDKCommand(sdk)
		return err
	},
}

func init() {
	rootCmd.AddCommand(updateSDKCmd)
}

func theUpdateSDKCommand(sdk string) error {
	println("Updating SDK repo...")
	absolutePathOfSDK, err := filepath.Abs(sdk)
	if err != nil {
		return fmt.Errorf("failed to get the directory of SDK: %v", err)
	}
	err = changeDir(absolutePathOfSDK)
	if err != nil {
		return fmt.Errorf("failed to change directory to %s: %v", absolutePathOfSDK, err)
	}
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get the current working directory: %v", err)
	}
	wt, err := repo.Get(cwd)
	if err != nil {
		return fmt.Errorf("failed to get the working tree: %v", err)
	}
	initialBranch, err = wt.Branch()
	if err != nil {
		return fmt.Errorf("failed to get the current branch: %v", err)
	}
	currentMajorVersion, err := findNextMajorVersionNumber(wt)
	majorVersion = currentMajorVersion + 1
	printf("Next major version: %d\n", majorVersion)
	vprintf("Checking out to latest branch in %s\n", cwd)
	err = wt.Checkout(master)
	if err != nil {
		return fmt.Errorf("checkout failed: %v", err)
	}
	err = wt.Pull(upstream, master)
	if err != nil {
		return fmt.Errorf("pull failed: %v", err)
	}
	vprintf("Checking out to new branch based on %s", master)
	branchName := fmt.Sprintf(branchPattern, majorVersion)
	err = createNewBranch(wt, branchName)
	if err != nil {
		return fmt.Errorf("checkout failed: %v", err)
	}
	majorBranchName = &branchName
	return nil
}

func findNextMajorVersionNumber(wt repo.WorkingTree) (int, error) {
	tags, err := wt.ListTags("v*")
	if err != nil {
		return 0, fmt.Errorf("failed to list tags: %v", err)
	}
	number := 0
	for _, tag := range tags {
		matches := pattern.FindStringSubmatch(tag)
		cc, err := strconv.ParseInt(matches[1], 10, 32)
		c := int(cc)
		if err == nil && c > number {
			number = c
		}
	}
	return number, nil
}
