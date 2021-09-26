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
	"os/signal"
	"path/filepath"
	"regexp"

	"github.com/Azure/azure-sdk-for-go/tools/apidiff/repo"
	"github.com/spf13/cobra"
)

const (
	latest        = "latest"
	master        = "master"
	specUpstream  = "origin"
	branchPattern = "major-version-release-v%d.0.0"
	readme        = "readme.md"
)

// flags
var upstream string
var quietFlag bool
var debugFlag bool
var verboseFlag bool
var thread int

// global variables
var initialBranch string
var pattern *regexp.Regexp
var majorVersion int
var majorBranchName *string

var rootCmd = &cobra.Command{
	Use:   "major-updater <SDK dir> <specification dir>",
	Short: "Do a whole procedure of monthly regular major update",
	Long:  `This tool will execute a procedure of releasing a new major update of the azure-sdk-for-go`,
	Args: func(cmd *cobra.Command, args []string) error {
		return cobra.ExactArgs(2)(cmd, args)
	},
	RunE: func(cmd *cobra.Command, args []string) error {
		absSDK, absSpec, err := absPaths(args[0], args[1])
		captureSigInt(absSDK)
		if err != nil {
			return err
		}
		err = theCommand(absSDK, absSpec)
		if err != nil {
			deleteMajorBranch(absSDK)
		}
		return err
	},
}

func init() {
	pattern = regexp.MustCompile(`^v([0-9]+)\..*$`)
	rootCmd.PersistentFlags().StringVar(&upstream, "upstream", "origin", "specify the upstream of the SDK repo")
	rootCmd.PersistentFlags().IntVarP(&thread, "thread", "t", 4, "thread count when executing autorest")
	rootCmd.PersistentFlags().BoolVarP(&quietFlag, "quiet", "q", false, "quiet output")
	rootCmd.PersistentFlags().BoolVarP(&debugFlag, "debug", "d", false, "debug output")
	rootCmd.PersistentFlags().BoolVarP(&verboseFlag, "verbose", "v", false, "verbose output")
}

func captureSigInt(absSDK string) {
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	go func() {
		for sig := range c {
			// sig is a ^C, handle it
			printf("Receiving %v signal, aborting...", sig)
			deleteMajorBranch(absSDK)
			os.Exit(1)
		}
	}()
}

// Execute executes the specified command.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func theCommand(absSDK, absSpec string) error {
	verboseStatus(absSDK, absSpec)
	var err error
	if err = theUpdateSDKCommand(absSDK); err != nil {
		return fmt.Errorf("failed to update SDK repo: %v", err)
	}
	if err = theDepCommand(); err != nil {
		return fmt.Errorf("failed to run dep: %v", err)
	}
	if err = theUpdateSpecsCommand(absSpec); err != nil {
		return fmt.Errorf("failed to update specs repo: %v", err)
	}
	if err = theAutorestCommand(absSDK, absSpec); err != nil {
		return fmt.Errorf("failed to execute autorest: %v", err)
	}
	if err = theAfterscriptsCommand(absSDK); err != nil {
		return fmt.Errorf("failed to execute afterscripts: %v", err)
	}
	return nil
}

func absPaths(sdk, spec string) (string, string, error) {
	absSDK, err := filepath.Abs(sdk)
	if err != nil {
		return "", "", fmt.Errorf("failed to get directory of SDK: %v", err)
	}
	absSpec, err := filepath.Abs(spec)
	if err != nil {
		return "", "", fmt.Errorf("failed to get directory of specification: %v", err)
	}
	return absSDK, absSpec, nil
}

func verboseStatus(sdk, spec string) {
	if verboseFlag {
		vprintf("SDK directory: %s\nSpecification directory: %s\n", sdk, spec)
	}
}

func createNewBranch(wt repo.WorkingTree, name string) error {
	vprintf("creating branch %s\n", name)
	err := wt.CreateAndCheckout(name)
	return err
}

func deleteMajorBranch(sdk string) {
	if majorBranchName == nil {
		return
	}
	changeDir(sdk)
	wt, err := repo.Get(sdk)
	if err != nil {
		return
	}
	wt.Checkout(latest)
	wt.DeleteBranch(*majorBranchName)
}

func changeDir(path string) error {
	err := os.Chdir(path)
	if err != nil {
		return fmt.Errorf("failed to change directory to %s: %v", path, err)
	}
	return nil
}
