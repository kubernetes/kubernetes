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
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/Azure/azure-sdk-for-go/tools/apidiff/exports"
	"github.com/Azure/azure-sdk-for-go/tools/apidiff/repo"
	"github.com/Azure/azure-sdk-for-go/tools/apidiff/report"
	"github.com/spf13/cobra"
)

const apiDirSuffix = "api"

var packagesCmd = &cobra.Command{
	Use:   "packages <package search dir> (<base commit> <target commit(s)>) | (<commit sequence>)",
	Short: "Generates a report for all packages under the specified directory containing the delta between commits.",
	Long: `The packages command generates a report for all of the packages under the directory specified in <package dir>.
Commits can be specified as either a base and one or more target commits or a sequence of commits.
For a base/target pair each target commit is compared against the base commit.
For a commit sequence each commit N in the sequence is compared against commit N+1.
Commit sequences must be comma-delimited.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		rpt, err := thePackagesCmd(args)
		if err != nil {
			return err
		}
		err = printReport(rpt)
		if err != nil {
			return err
		}
		evalReportStatus(rpt)
		return nil
	},
}

func init() {
	rootCmd.AddCommand(packagesCmd)
}

// ExecPackagesCmd is the programmatic interface for the packages command.
func ExecPackagesCmd(pkgDir string, commitSeq string, flags CommandFlags) (CommitPkgsReport, error) {
	flags.apply()
	return thePackagesCmd([]string{pkgDir, commitSeq})
}

// split into its own func as we can't call os.Exit from it (the defer won't get executed)
func thePackagesCmd(args []string) (rpt CommitPkgsReport, err error) {
	cloneRepo, err := processArgsAndClone(args)
	if err != nil {
		return
	}

	rpt.CommitsReports = map[string]pkgsReport{}
	worker := func(rootDir string, cloneRepo repo.WorkingTree, baseCommit, targetCommit string) error {
		vprintf("generating diff between %s and %s\n", baseCommit, targetCommit)
		// get for lhs
		dprintf("checking out base commit %s and gathering exports\n", baseCommit)
		lhs, err := getRepoContentForCommit(cloneRepo, rootDir, baseCommit)
		if err != nil {
			return err
		}

		// get for rhs
		dprintf("checking out target commit %s and gathering exports\n", targetCommit)
		var rhs repoContent
		rhs, err = getRepoContentForCommit(cloneRepo, rootDir, targetCommit)
		if err != nil {
			return err
		}
		r := getPkgsReport(lhs, rhs)
		rpt.updateAffectedPackages(targetCommit, r)
		if r.hasBreakingChanges() {
			rpt.BreakingChanges = append(rpt.BreakingChanges, targetCommit)
		}
		rpt.CommitsReports[fmt.Sprintf("%s:%s", baseCommit, targetCommit)] = r
		return nil
	}

	err = generateReports(args, cloneRepo, worker)
	if err != nil {
		return
	}

	return
}

func getRepoContentForCommit(wt repo.WorkingTree, dir, commit string) (r repoContent, err error) {
	err = wt.Checkout(commit)
	if err != nil {
		err = fmt.Errorf("failed to check out commit '%s': %s", commit, err)
		return
	}

	pkgDirs := []string{}
	err = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		var skipDir error
		if info.IsDir() {
			// check if leaf dir
			fi, err := ioutil.ReadDir(path)
			if err != nil {
				return err
			}
			hasSubDirs := false
			for _, f := range fi {
				// check if this is the interfaces subdir, if it is don't count it as a subdir
				if f.IsDir() && f.Name() != filepath.Base(path)+apiDirSuffix {
					hasSubDirs = true
					break
				}
			}
			if !hasSubDirs {
				pkgDirs = append(pkgDirs, path)
				// skip any dirs under us (i.e. interfaces subdir)
				skipDir = filepath.SkipDir
			}
		}
		return skipDir
	})
	if err != nil {
		return
	}
	if debugFlag {
		fmt.Println("found the following package directories")
		for _, d := range pkgDirs {
			fmt.Printf("\t%s\n", d)
		}
	}

	r, err = getExportsForPackages(wt.Root(), pkgDirs)
	if err != nil {
		err = fmt.Errorf("failed to get exports for commit '%s': %s", commit, err)
	}
	return
}

// contains repo content, it's structured as "package path":content
type repoContent map[string]exports.Content

// returns repoContent based on the provided slice of package directories
func getExportsForPackages(root string, pkgDirs []string) (repoContent, error) {
	exps := repoContent{}
	for _, pkgDir := range pkgDirs {
		dprintf("getting exports for %s\n", pkgDir)
		// pkgDir = "C:\Users\somebody\AppData\Local\Temp\apidiff-1529437978\services\addons\mgmt\2017-05-15\addons"
		// convert to package path "github.com/Azure/azure-sdk-for-go/services/analysisservices/mgmt/2016-05-16/analysisservices"
		pkgPath := strings.Replace(pkgDir, root, "github.com/Azure/azure-sdk-for-go", -1)
		pkgPath = strings.Replace(pkgPath, string(os.PathSeparator), "/", -1)
		if _, ok := exps[pkgPath]; ok {
			return nil, fmt.Errorf("duplicate package: %s", pkgPath)
		}
		exp, err := exports.Get(pkgDir)
		if err != nil {
			return nil, err
		}
		exps[pkgPath] = exp
	}
	return exps, nil
}

// contains a collection of packages
type pkgsList []string

// contains a collection of package reports, it's structured as "package path":pkgReport
type modifiedPackages map[string]report.Package

// CommitPkgsReport represents a collection of reports, one for each commit hash.
type CommitPkgsReport struct {
	AffectedPackages map[string]pkgsList   `json:"affectedPackages"`
	BreakingChanges  []string              `json:"breakingChanges,omitempty"`
	CommitsReports   map[string]pkgsReport `json:"deltas"`
}

// IsEmpty returns true if the report contains no data.
func (c CommitPkgsReport) IsEmpty() bool {
	for _, r := range c.CommitsReports {
		if !r.isEmpty() {
			return false
		}
	}
	return true
}

// HasBreakingChanges returns true if the report contains breaking changes.
func (c CommitPkgsReport) HasBreakingChanges() bool {
	for _, r := range c.CommitsReports {
		if r.hasBreakingChanges() {
			return true
		}
	}
	return false
}

// HasAdditiveChanges returns true if the package contains additive changes.
func (c CommitPkgsReport) HasAdditiveChanges() bool {
	for _, r := range c.CommitsReports {
		if r.hasAdditiveChanges() {
			return true
		}
	}
	return false
}

// updates the collection of affected packages with the packages that were touched in the specified commit
func (c *CommitPkgsReport) updateAffectedPackages(commit string, r pkgsReport) {
	if c.AffectedPackages == nil {
		c.AffectedPackages = map[string]pkgsList{}
	}

	for _, pkg := range r.AddedPackages {
		c.AffectedPackages[commit] = append(c.AffectedPackages[commit], pkg)
	}

	for pkgName := range r.ModifiedPackages {
		c.AffectedPackages[commit] = append(c.AffectedPackages[commit], pkgName)
	}

	for _, pkg := range r.RemovedPackages {
		c.AffectedPackages[commit] = append(c.AffectedPackages[commit], pkg)
	}
}

// represents a complete report of added, removed, and modified packages
type pkgsReport struct {
	AddedPackages      pkgsList         `json:"added,omitempty"`
	ModifiedPackages   modifiedPackages `json:"modified,omitempty"`
	RemovedPackages    pkgsList         `json:"removed,omitempty"`
	modPkgHasAdditions bool
	modPkgHasBreaking  bool
}

// returns true if the package report contains breaking changes
func (r pkgsReport) hasBreakingChanges() bool {
	return len(r.RemovedPackages) > 0 || r.modPkgHasBreaking
}

// returns true if the package report contains additive changes
func (r pkgsReport) hasAdditiveChanges() bool {
	return len(r.AddedPackages) > 0 || r.modPkgHasAdditions
}

// returns true if the report contains no data
func (r pkgsReport) isEmpty() bool {
	return len(r.AddedPackages) == 0 && len(r.ModifiedPackages) == 0 && len(r.RemovedPackages) == 0
}

// generates a pkgsReport based on the delta between lhs and rhs
func getPkgsReport(lhs, rhs repoContent) pkgsReport {
	rpt := pkgsReport{}

	if !onlyBreakingChangesFlag {
		rpt.AddedPackages = getPkgsList(lhs, rhs)
	}
	if !onlyAdditionsFlag {
		rpt.RemovedPackages = getPkgsList(rhs, lhs)
	}

	// diff packages
	for rhsPkg, rhsCnt := range rhs {
		if _, ok := lhs[rhsPkg]; !ok {
			continue
		}
		if r := report.Generate(lhs[rhsPkg], rhsCnt, onlyBreakingChangesFlag, onlyAdditionsFlag); !r.IsEmpty() {
			if r.HasBreakingChanges() {
				rpt.modPkgHasBreaking = true
			}
			if r.HasAdditiveChanges() {
				rpt.modPkgHasAdditions = true
			}
			// only add an entry if the report contains data
			if rpt.ModifiedPackages == nil {
				rpt.ModifiedPackages = modifiedPackages{}
			}
			rpt.ModifiedPackages[rhsPkg] = r
		}
	}

	return rpt
}

// returns a list of packages in rhs that aren't in lhs
func getPkgsList(lhs, rhs repoContent) pkgsList {
	list := pkgsList{}
	for rhsPkg := range rhs {
		if _, ok := lhs[rhsPkg]; !ok {
			list = append(list, rhsPkg)
		}
	}
	return list
}
