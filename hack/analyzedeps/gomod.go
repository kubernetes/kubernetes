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
	"fmt"
	"os"
	"strings"

	"golang.org/x/mod/modfile"
)

type GoModInfo struct {
	Deps           map[string]string
	Indirect       map[string]bool
	StagingModules map[string]bool
}

// parseGoMod parses the go.mod file at the given path and extracts dependencies and staging modules.
func parseGoMod(goModPath string) (*GoModInfo, error) {
	goModContent, err := os.ReadFile(goModPath)
	if err != nil {
		return nil, fmt.Errorf("error reading go.mod: %w", err)
	}

	file, err := modfile.Parse(goModPath, goModContent, nil)
	if err != nil {
		return nil, fmt.Errorf("error parsing go.mod: %w", err)
	}

	// Identify staging/internal modules from replace directives
	stagingModules := make(map[string]bool)
	for _, rep := range file.Replace {
		// Any replacement with a local path under staging is an internal staging module
		if strings.HasPrefix(rep.New.Path, "./staging/") || strings.HasPrefix(rep.New.Path, "../staging/") || strings.Contains(rep.New.Path, "/staging/src/") {
			stagingModules[rep.Old.Path] = true
		}
	}

	goModDeps := make(map[string]string)
	goModIndirect := make(map[string]bool)

	for _, req := range file.Require {
		modPath := req.Mod.Path
		// Skip staging modules and main module
		if stagingModules[modPath] || modPath == "k8s.io/kubernetes" {
			continue
		}
		goModDeps[modPath] = req.Mod.Version
		goModIndirect[modPath] = req.Indirect
	}

	return &GoModInfo{
		Deps:           goModDeps,
		Indirect:       goModIndirect,
		StagingModules: stagingModules,
	}, nil
}
