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
	"debug/buildinfo"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
)

type ModuleInfo struct {
	Path    string
	Version string
	Main    bool
}

type Package struct {
	ImportPath string
	Standard   bool
	Module     *ModuleInfo
}

func getListLocalScript() (string, error) {
	root, err := getRepoRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "hack", "analyzedeps", "list_local.sh"), nil
}

// findGoMod returns the absolute path to the repository's go.mod file.
func findGoMod() (string, error) {
	root, err := getRepoRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "go.mod"), nil
}

// scanLocalWorkspace runs the local go list script and reads package dependencies from its output.
func scanLocalWorkspace() (map[string]bool, error) {
	scriptPath, err := getListLocalScript()
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(scriptPath)
	cmd.Stderr = os.Stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdout pipe for list_local.sh: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start list_local.sh: %w", err)
	}

	productionDeps := make(map[string]bool)
	dec := json.NewDecoder(stdout)
	for dec.More() {
		var pkg Package
		if err := dec.Decode(&pkg); err != nil {
			return nil, fmt.Errorf("error decoding go list JSON stream: %w", err)
		}
		if pkg.Standard {
			continue
		}
		if pkg.Module != nil && !pkg.Module.Main {
			productionDeps[pkg.Module.Path] = true
		}
	}

	if err := cmd.Wait(); err != nil {
		return nil, fmt.Errorf("list_local.sh execution failed: %w", err)
	}
	return productionDeps, nil
}

// scanBinaries recursively scans the given directory for Go binaries and extracts their dependencies.
func scanBinaries(binariesDir string) (map[string]bool, error) {
	productionDeps := make(map[string]bool)

	err := filepath.WalkDir(binariesDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}

		info, err := buildinfo.ReadFile(path)
		if err != nil {
			return nil
		}

		for _, dep := range info.Deps {
			productionDeps[dep.Path] = true
		}

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("error scanning binaries: %w", err)
	}

	return productionDeps, nil
}
