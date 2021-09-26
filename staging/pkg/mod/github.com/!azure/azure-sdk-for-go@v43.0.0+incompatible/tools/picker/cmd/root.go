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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	apidiff "github.com/Azure/azure-sdk-for-go/tools/apidiff/cmd"
	"github.com/Azure/azure-sdk-for-go/tools/apidiff/repo"
	"github.com/spf13/cobra"
)

var dryRunFlag bool

var rootCmd = &cobra.Command{
	Use:   "picker <from> <to>",
	Short: "Cherry-picks commits with non-breaking changes between two branches.",
	Long: `This tool will find the list of commits in branch <from> that are not in
branch <to>, and for each commit found it will cherry-pick it into <to> if
the commit contains no breaking changes.  If a cherry-pick contains a merge
conflict the process will pause so the conflicts can be resolved.
NOTE: running this tool will modify your working tree!`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return theCommand(args)
	},
}

func init() {
	rootCmd.PersistentFlags().BoolVarP(&dryRunFlag, "dryrun", "d", false, "reports what the tool would have done without actually doing it")
}

// Execute executes the specified command.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(-1)
	}
}

func theCommand(args []string) error {
	if len(args) < 2 {
		return errors.New("not enough arguments were supplied")
	}

	from := args[0]
	to := args[1]

	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get the current working directory: %v", err)
	}

	wt, err := repo.Get(cwd)
	if err != nil {
		return fmt.Errorf("failed to get the working tree: %v", err)
	}

	if dryRunFlag {
		// dry run shouldn't modify state so ensure that the active branch matches from
		branch, err := wt.Branch()
		if err != nil {
			return err
		}
		if branch != from {
			return fmt.Errorf("dry run failed, current branch '%s' doesn't match from branch '%s'", branch, from)
		}
	}
	// checkout "from" branch
	fmt.Printf("checking out branch '%s' to get list of candidate commits for cherry-picking\n", from)
	err = wt.Checkout(from)
	if err != nil {
		return fmt.Errorf("checkout failed: %v", err)
	}

	commits, err := wt.Cherry(to)
	if err != nil {
		return fmt.Errorf("the command 'git cherry' failed: %v", err)
	}

	fmt.Printf("found %v commits, calculating breaking changes...\n", len(commits))

	// generate report to find the breaking changes
	report, err := apidiff.ExecPackagesCmd(filepath.Join(wt.Root(), "services"),
		fmt.Sprintf("%s~1,%s", commits[0].Hash, strings.Join(func() []string {
			hashes := []string{}
			for _, commit := range commits {
				hashes = append(hashes, commit.Hash)
			}
			return hashes
		}(), ",")),
		apidiff.CommandFlags{
			SuppressReport: true,
			Verbose:        true,
		})
	if err != nil {
		return fmt.Errorf("failed to obtain the breaking changes report: %v", err)
	}

	forPicking := pruneCommits(commits, report)
	if len(forPicking) == 0 {
		fmt.Println("didn't find any commits to cherry-pick")
		return nil
	}

	fmt.Println("will cherry-pick the following commits:")
	for _, commit := range forPicking {
		fmt.Printf("\t%s\n", commit)
	}

	if dryRunFlag {
		return nil
	}

	// now cherry-pick the commits
	fmt.Printf("checking out branch '%s' to begin cherry-picking\n", to)
	err = wt.Checkout(to)
	if err != nil {
		return fmt.Errorf("checkout failed: %v", err)
	}

	for _, commit := range forPicking {
		fmt.Printf("cherry-picking commit %s\n", commit)
		err = wt.CherryPick(commit)
		if err != nil {
			fmt.Printf("there was an error: %v\n", err)
			fmt.Println("if this is due to a merge conflict fix the conflict then press ENTER to continue cherry-picking, else press CTRL-C to abort")
			dummy := ""
			fmt.Scanln(&dummy)
		}
	}
	return nil
}

// performs a linear search of ss, looking for s
func contains(ss []string, s string) bool {
	for i := range ss {
		if ss[i] == s {
			return true
		}
	}
	return false
}

// removes commits and decendents thereof that contain breaking changes
func pruneCommits(commits []repo.CherryCommit, report apidiff.CommitPkgsReport) []string {
	pkgsToSkip := map[string]string{}
	forPicking := []string{}
	for _, commit := range commits {
		if commit.Found {
			// if this commit was found in the target branch skip it
			fmt.Printf("skipping %s as it was found in the target branch\n", commit.Hash)
			continue
		}
		if contains(report.BreakingChanges, commit.Hash) {
			fmt.Printf("omitting %s as it contains breaking changes\n", commit.Hash)
			// add the affected packages to the collection of packages to skip
			for _, pkg := range report.AffectedPackages[commit.Hash] {
				if _, ok := pkgsToSkip[pkg]; !ok {
					pkgsToSkip[pkg] = commit.Hash
				}
			}
			continue
		}

		// check the packages impacted by this commit, if any of them are in
		// the list of packages to skip then this commit can't be cherry-picked
		include := true
		for _, pkg := range report.AffectedPackages[commit.Hash] {
			if _, ok := pkgsToSkip[pkg]; ok {
				include = false
				fmt.Printf("omitting %s as it's an aggregate of commit %s that includes breaking changes\n", commit.Hash, pkgsToSkip[pkg])
				break
			}
		}

		if include {
			forPicking = append(forPicking, commit.Hash)
		}
	}
	return forPicking
}
