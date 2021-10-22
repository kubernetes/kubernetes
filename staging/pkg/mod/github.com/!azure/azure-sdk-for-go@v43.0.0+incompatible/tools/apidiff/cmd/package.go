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

	"github.com/Azure/azure-sdk-for-go/tools/apidiff/exports"
	"github.com/Azure/azure-sdk-for-go/tools/apidiff/repo"
	"github.com/Azure/azure-sdk-for-go/tools/apidiff/report"
	"github.com/spf13/cobra"
)

var (
	dirMode    bool
	asMarkdown bool
)

var packageCmd = &cobra.Command{
	Use:   "package <package dir> (<base commit> <target commit(s)>) | (<commit sequence>)",
	Short: "Generates a report for the package in the specified directory containing the delta between commits.",
	Long: `The package command generates a report for the package in the directory specified in <package dir>.
Commits can be specified as either a base and one or more target commits or a sequence of commits.
For a base/target pair each target commit is compared against the base commit.
For a commit sequence each commit N in the sequence is compared against commit N+1.
Commit sequences must be comma-delimited.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		rpt, err := thePackageCmd(args)
		if err != nil {
			return err
		}
		evalReportStatus(rpt)
		return nil
	},
}

// split into its own func as we can't call os.Exit from it (the defer won't get executed)
func thePackageCmd(args []string) (rs reportStatus, err error) {
	if dirMode {
		return packageCmdDirMode(args)
	}

	cloneRepo, err := processArgsAndClone(args)
	if err != nil {
		return
	}

	var rpt commitPkgReport
	rpt.CommitsReports = map[string]report.Package{}
	worker := func(pkgDir string, cloneRepo repo.WorkingTree, baseCommit, targetCommit string) error {
		// lhs
		vprintf("checking out base commit %s and gathering exports\n", baseCommit)
		var lhs exports.Content
		lhs, err = getContentForCommit(cloneRepo, pkgDir, baseCommit)
		if err != nil {
			return err
		}

		// rhs
		vprintf("checking out target commit %s and gathering exports\n", targetCommit)
		var rhs exports.Content
		rhs, err = getContentForCommit(cloneRepo, pkgDir, targetCommit)
		if err != nil {
			return err
		}
		r := report.Generate(lhs, rhs, onlyBreakingChangesFlag, onlyAdditionsFlag)
		if r.HasBreakingChanges() {
			rpt.BreakingChanges = append(rpt.BreakingChanges, targetCommit)
		}
		rpt.CommitsReports[fmt.Sprintf("%s:%s", baseCommit, targetCommit)] = r
		return nil
	}

	err = generateReports(args, cloneRepo, worker)
	if err != nil {
		return
	}

	err = printReport(rpt)
	return
}

func init() {
	packageCmd.PersistentFlags().BoolVarP(&dirMode, "directories", "i", false, "compares packages in two different directories")
	packageCmd.PersistentFlags().BoolVarP(&asMarkdown, "markdown", "m", false, "emits the report in markdown format")
	rootCmd.AddCommand(packageCmd)
}

func getContentForCommit(wt repo.WorkingTree, dir, commit string) (cnt exports.Content, err error) {
	err = wt.Checkout(commit)
	if err != nil {
		err = fmt.Errorf("failed to check out commit '%s': %s", commit, err)
		return
	}

	cnt, err = exports.Get(dir)
	if err != nil {
		err = fmt.Errorf("failed to get exports for commit '%s': %s", commit, err)
	}
	return
}

func packageCmdDirMode(args []string) (rs reportStatus, err error) {
	if len(args) != 2 {
		return nil, errors.New("directories mode requires two arguments")
	}
	lhs, err := exports.Get(args[0])
	if err != nil {
		return nil, fmt.Errorf("failed to get exports for package '%s': %s", args[0], err)
	}
	rhs, err := exports.Get(args[1])
	if err != nil {
		return nil, fmt.Errorf("failed to get exports for package '%s': %s", args[1], err)
	}
	r := report.Generate(lhs, rhs, onlyBreakingChangesFlag, onlyAdditionsFlag)
	if asMarkdown && !suppressReport {
		println(r.ToMarkdown())
	} else {
		err = printReport(r)
	}
	return r, err
}
