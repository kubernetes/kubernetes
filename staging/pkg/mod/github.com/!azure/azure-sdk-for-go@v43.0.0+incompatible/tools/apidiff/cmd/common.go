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
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Azure/azure-sdk-for-go/tools/apidiff/ioext"
	"github.com/Azure/azure-sdk-for-go/tools/apidiff/repo"
	"github.com/Azure/azure-sdk-for-go/tools/apidiff/report"
)

func printf(format string, a ...interface{}) {
	if !quietFlag {
		fmt.Printf(format, a...)
	}
}

func println(a ...interface{}) {
	if !quietFlag {
		fmt.Println(a...)
	}
}

func dprintf(format string, a ...interface{}) {
	if debugFlag {
		printf(format, a...)
	}
}

func dprintln(a ...interface{}) {
	if debugFlag {
		println(a...)
	}
}

func vprintf(format string, a ...interface{}) {
	if verboseFlag {
		printf(format, a...)
	}
}

func vprintln(a ...interface{}) {
	if verboseFlag {
		println(a...)
	}
}

// represents a collection of per-package reports, one for each commit hash
type commitPkgReport struct {
	BreakingChanges []string                  `json:"breakingChanges,omitempty"`
	CommitsReports  map[string]report.Package `json:"deltas"`
}

// returns true if the report contains no data
func (c commitPkgReport) IsEmpty() bool {
	for _, rpt := range c.CommitsReports {
		if !rpt.IsEmpty() {
			return false
		}
	}
	return true
}

// returns true if the report contains breaking changes
func (c commitPkgReport) hasBreakingChanges() bool {
	for _, r := range c.CommitsReports {
		if r.HasBreakingChanges() {
			return true
		}
	}
	return false
}

// returns true if the report contains additive changes
func (c commitPkgReport) hasAdditiveChanges() bool {
	for _, r := range c.CommitsReports {
		if r.HasAdditiveChanges() {
			return true
		}
	}
	return false
}

type reportInfo interface {
	IsEmpty() bool
}

func printReport(r reportInfo) error {
	if r.IsEmpty() {
		println("no changes were found")
		return nil
	}

	if !suppressReport {
		b, err := json.MarshalIndent(r, "", "  ")
		if err != nil {
			return fmt.Errorf("failed to marshal report: %v", err)
		}
		println(string(b))
	}
	return nil
}

func processArgsAndClone(args []string) (cln repo.WorkingTree, err error) {
	if onlyAdditionsFlag && onlyBreakingChangesFlag {
		err = errors.New("flags 'additions' and 'breakingchanges' are mutually exclusive")
		return
	}

	// there should be at minimum two args, a directory and a
	// sequence of commits, i.e. "d:\foo 1,2,3".  else a directory
	// and two commits, i.e. "d:\foo 1 2" or "d:\foo 1 2,3"
	if len(args) < 2 {
		err = errors.New("not enough args were supplied")
		return
	}

	// here args[1] should be a comma-delimited list of commits
	if len(args) == 2 && strings.Index(args[1], ",") < 0 {
		err = errors.New("expected a comma-delimited list of commits")
		return
	}

	dir := args[0]
	dir, err = filepath.Abs(dir)
	if err != nil {
		err = fmt.Errorf("failed to convert path '%s' to absolute path: %v", dir, err)
		return
	}

	src, err := repo.Get(dir)
	if err != nil {
		err = fmt.Errorf("failed to get repository: %v", err)
		return
	}

	tempRepoDir := filepath.Join(os.TempDir(), fmt.Sprintf("apidiff-%v", time.Now().Unix()))
	if copyRepoFlag {
		vprintf("copying '%s' into '%s'...\n", src.Root(), tempRepoDir)
		err = ioext.CopyDir(src.Root(), tempRepoDir)
		if err != nil {
			err = fmt.Errorf("failed to copy repo: %v", err)
			return
		}
		cln, err = repo.Get(tempRepoDir)
		if err != nil {
			err = fmt.Errorf("failed to get copied repo: %v", err)
			return
		}
	} else {
		vprintf("cloning '%s' into '%s'...\n", src.Root(), tempRepoDir)
		cln, err = src.Clone(tempRepoDir)
		if err != nil {
			err = fmt.Errorf("failed to clone repository: %v", err)
			return
		}
	}

	// fix up pkgDir to the clone
	args[0] = strings.Replace(dir, src.Root(), cln.Root(), 1)

	return
}

type reportGenFunc func(dir string, cln repo.WorkingTree, baseCommit, targetCommit string) error

func generateReports(args []string, cln repo.WorkingTree, fn reportGenFunc) error {
	defer func() {
		// delete clone
		vprintln("cleaning up clone")
		err := os.RemoveAll(cln.Root())
		if err != nil {
			vprintf("failed to delete temp repo: %v\n", err)
		}
	}()

	var commits []string

	// if the commits are specified as 1 2,3,4 then it means that commit 1 is
	// always the base commit and to compare it against each target commit in
	// the sequence.  however if it's specifed as 1,2,3,4 then the base commit
	// is relative to the iteration, i.e. compare 1-2, 2-3, 3-4.
	fixedBase := true
	if len(args) == 3 {
		commits = make([]string, 2, 2)
		commits[0] = args[1]
		commits[1] = args[2]
	} else {
		commits = strings.Split(args[1], ",")
		fixedBase = false
	}

	for i := 0; i+1 < len(commits); i++ {
		baseCommit := commits[i]
		if fixedBase {
			baseCommit = commits[0]
		}
		targetCommit := commits[i+1]

		err := fn(args[0], cln, baseCommit, targetCommit)
		if err != nil {
			return err
		}
	}
	return nil
}

type reportStatus interface {
	HasBreakingChanges() bool
	HasAdditiveChanges() bool
}

// compares report status with the desired report options (breaking/additions)
// to determine if the program should terminate with a non-zero exit code.
func evalReportStatus(r reportStatus) {
	if onlyBreakingChangesFlag && r.HasBreakingChanges() {
		os.Exit(1)
	}
	if onlyAdditionsFlag && !r.HasAdditiveChanges() {
		os.Exit(1)
	}
}
