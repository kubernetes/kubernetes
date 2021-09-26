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
	"sort"
	"strings"

	"github.com/spf13/cobra"
)

var changelogCmd = &cobra.Command{
	Use:   "changelog <package search dir> <base commit> <target commit>",
	Short: "Generates a CHANGELOG report in markdown format for the packages under the specified directory.",
	Long: `The changelog command generates a CHANGELOG for all of the packages under the directory specified in <package dir>.
A table for added, removed, updated, and breaking changes will be created as required.`,
	Args: func(cmd *cobra.Command, args []string) error {
		// there should be exactly three args, a directory and two commit hashes
		if err := cobra.ExactArgs(3)(cmd, args); err != nil {
			return err
		}
		if strings.Index(args[2], ",") > -1 {
			return errors.New("sequence of target commits is not supported")
		}
		return nil
	},
	RunE: func(cmd *cobra.Command, args []string) error {
		return theChangelogCmd(args)
	},
}

func init() {
	rootCmd.AddCommand(changelogCmd)
}

func theChangelogCmd(args []string) error {
	// TODO: refactor so that we don't depend on the packages command
	rpt, err := thePackagesCmd(args)
	if err != nil {
		return err
	}
	if rpt.IsEmpty() {
		return nil
	}

	// there should only be one report, the delta between the base and target commits
	if len(rpt.CommitsReports) > 1 {
		panic("expected only one report")
	}
	for _, cr := range rpt.CommitsReports {
		reportAddedPkgs(cr)
		reportUpdatedPkgs(cr)
		reportBreakingPkgs(cr)
		reportRemovedPkgs(cr)
	}
	return nil
}

func reportAddedPkgs(pr pkgsReport) {
	if len(pr.AddedPackages) == 0 {
		return
	}
	fmt.Printf("### New Packages\n\n")
	createTable(createTableRows(pr.AddedPackages))
}

func reportUpdatedPkgs(pr pkgsReport) {
	if !pr.modPkgHasAdditions {
		return
	}
	fmt.Printf("### Updated Packages\n\n")
	updated := []string{}
	for pkgName, pkgRpt := range pr.ModifiedPackages {
		if pkgRpt.HasAdditiveChanges() && !pkgRpt.HasBreakingChanges() {
			updated = append(updated, pkgName)
		}
	}
	createTable(createTableRows(updated))
}

func reportBreakingPkgs(pr pkgsReport) {
	if !pr.modPkgHasBreaking {
		return
	}
	fmt.Printf("### BreakingChanges\n\n")
	breaking := []string{}
	for pkgName, pkgRpt := range pr.ModifiedPackages {
		if pkgRpt.HasBreakingChanges() {
			breaking = append(breaking, pkgName)
		}
	}
	createTable(createTableRows(breaking))
}

func reportRemovedPkgs(pr pkgsReport) {
	if len(pr.RemovedPackages) == 0 {
		return
	}
	fmt.Printf("### Removed Packages\n\n")
	createTable(createTableRows(pr.RemovedPackages))
}

type tableRow struct {
	pkgName     string
	apiVersions []string
}

func createTableRows(pkgs []string) []tableRow {
	entries := map[string][]string{}
	for _, pkg := range pkgs {
		// contains entries like "github.com/Azure/azure-sdk-for-go/services/consumption/mgmt/2018-08-31/consumption"
		i := strings.LastIndex(pkg, "/")
		pkgName := pkg[i+1:]
		j := strings.LastIndex(pkg[:i], "/")
		apiVer := pkg[j+1 : i]
		if apis, ok := entries[pkgName]; ok {
			entries[pkgName] = append(apis, apiVer)
		} else {
			entries[pkgName] = []string{apiVer}
		}
	}
	// convert the map to a slice of tableRows
	rows := []tableRow{}
	for pkgName, apiVers := range entries {
		sort.Strings(apiVers)
		tr := tableRow{
			pkgName:     pkgName,
			apiVersions: apiVers,
		}
		rows = append(rows, tr)
	}
	sort.Slice(rows, func(i, j int) bool {
		return rows[i].pkgName < rows[j].pkgName
	})
	return rows
}

func createTable(rows []tableRow) {
	fmt.Println("| Package Name | API Version |")
	fmt.Println("| -----------: | :---------: |")
	for _, row := range rows {
		fmt.Println("| " + row.pkgName + " | " + strings.Join(row.apiVersions, "<br/>") + " |")
	}
	fmt.Println()
}
