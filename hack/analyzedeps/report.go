/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

type AnalysisReport struct {
	TotalGoModDeps         int      `json:"totalGoModDeps"`
	TotalProductionDeps    int      `json:"totalProductionDeps"`
	TotalNonProductionDeps int      `json:"totalNonProductionDeps"`
	ProductionDeps         []string `json:"productionDeps"`
	NonProductionDeps      []string `json:"nonProductionDeps"`
}

// analyzeDependencies contrasts production dependencies with go.mod declarations.
func analyzeDependencies(productionDeps map[string]bool, modInfo *GoModInfo) (*AnalysisReport, error) {
	var productionList []string
	var nonProductionList []string
	var undeclaredList []string

	for path := range modInfo.Deps {
		if productionDeps[path] {
			productionList = append(productionList, path)
		} else {
			nonProductionList = append(nonProductionList, path)
		}
	}

	for path := range productionDeps {
		if _, ok := modInfo.Deps[path]; !ok {
			if !modInfo.StagingModules[path] && path != "k8s.io/kubernetes" {
				undeclaredList = append(undeclaredList, path)
			}
		}
	}

	if len(undeclaredList) > 0 {
		sort.Strings(undeclaredList)
		return nil, fmt.Errorf("found %d undeclared dependencies compiled into binaries: %s", len(undeclaredList), strings.Join(undeclaredList, ", "))
	}

	sort.Strings(productionList)
	sort.Strings(nonProductionList)

	if productionList == nil {
		productionList = []string{}
	}
	if nonProductionList == nil {
		nonProductionList = []string{}
	}

	return &AnalysisReport{
		TotalGoModDeps:         len(modInfo.Deps),
		TotalProductionDeps:    len(productionList),
		TotalNonProductionDeps: len(nonProductionList),
		ProductionDeps:         productionList,
		NonProductionDeps:      nonProductionList,
	}, nil
}

// printReport outputs the final formatted dependency analysis report as JSON.
func printReport(report *AnalysisReport) {
	out, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error encoding JSON report: %v\n", err)
		return
	}
	fmt.Println(string(out))
}
