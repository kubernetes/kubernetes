/*
Copyright 2017 The Kubernetes Authors.

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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	yaml "go.yaml.in/yaml/v2"
)

// Package is a subset of cmd/go.Package
type Package struct {
	Dir          string   `yaml:",omitempty"` // directory containing package sources
	ImportPath   string   `yaml:",omitempty"` // import path of package
	Imports      []string `yaml:",omitempty"` // import paths used by this package
	TestImports  []string `yaml:",omitempty"` // imports from TestGoFiles
	XTestImports []string `yaml:",omitempty"` // imports from XTestGoFiles
}

// ImportRestriction describes a set of allowable import
// trees for a tree of source code
type ImportRestriction struct {
	// BaseDir is the root of a package tree that is restricted by this
	// configuration, given as a relative path from the root of the repository.
	// This is a directory which `go list` might not consider a package (if it
	// has not .go files)
	BaseDir string `yaml:"baseImportPath"`
	// IgnoredSubTrees are roots of sub-trees of the BaseDir for which we do
	// not want to enforce any import restrictions whatsoever, given as
	// relative paths from the root of the repository.
	IgnoredSubTrees []string `yaml:"ignoredSubTrees,omitempty"`
	// AllowedImports are roots of package trees that are allowed to be
	// imported from the BaseDir, given as paths that would be used in a Go
	// import statement.
	AllowedImports []string `yaml:"allowedImports"`
	// ExcludeTests will skip checking test dependencies.
	ExcludeTests bool `yaml:"excludeTests"`
}

// ForbiddenImportsFor determines all of the forbidden
// imports for a package given the import restrictions
func (i *ImportRestriction) ForbiddenImportsFor(pkg Package) ([]string, error) {
	if restricted, err := i.isRestrictedDir(pkg.Dir); err != nil {
		return []string{}, err
	} else if !restricted {
		return []string{}, nil
	}

	return i.forbiddenImportsFor(pkg), nil
}

// isRestrictedDir determines if the source directory has
// any restrictions placed on it by this configuration.
// A path will be restricted if:
//   - it falls under the base import path
//   - it does not fall under any of the ignored sub-trees
func (i *ImportRestriction) isRestrictedDir(dir string) (bool, error) {
	if under, err := isPathUnder(i.BaseDir, dir); err != nil {
		return false, err
	} else if !under {
		return false, nil
	}

	for _, ignored := range i.IgnoredSubTrees {
		if under, err := isPathUnder(ignored, dir); err != nil {
			return false, err
		} else if under {
			return false, nil
		}
	}

	return true, nil
}

// isPathUnder determines if path is under base
func isPathUnder(base, path string) (bool, error) {
	absBase, err := filepath.Abs(base)
	if err != nil {
		return false, err
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		return false, err
	}

	relPath, err := filepath.Rel(absBase, absPath)
	if err != nil {
		return false, err
	}

	// if path is below base, the relative path
	// from base to path will not start with `../`
	return !strings.HasPrefix(relPath, ".."), nil
}

// forbiddenImportsFor determines all of the forbidden
// imports for a package given the import restrictions
// and returns a deduplicated list of them
func (i *ImportRestriction) forbiddenImportsFor(pkg Package) []string {
	forbiddenImportSet := map[string]struct{}{}
	imports := pkg.Imports
	if !i.ExcludeTests {
		imports = append(imports, append(pkg.TestImports, pkg.XTestImports...)...)
	}
	for _, imp := range imports {
		if i.isForbidden(imp) {
			forbiddenImportSet[imp] = struct{}{}
		}
	}

	var forbiddenImports []string
	for imp := range forbiddenImportSet {
		forbiddenImports = append(forbiddenImports, imp)
	}
	return forbiddenImports
}

// isForbidden determines if an import is forbidden,
// which is true when the import is:
//   - of a package under the rootPackage
//   - is not of the base import path or a sub-package of it
//   - is not of an allowed path or a sub-package of one
func (i *ImportRestriction) isForbidden(imp string) bool {
	importsBelowRoot := strings.HasPrefix(imp, rootPackage)
	importsBelowBase := strings.HasPrefix(imp, i.BaseDir)
	importsAllowed := false
	for _, allowed := range i.AllowedImports {
		exactlyImportsAllowed := imp == allowed
		importsBelowAllowed := strings.HasPrefix(imp, fmt.Sprintf("%s/", allowed))
		importsAllowed = importsAllowed || (importsBelowAllowed || exactlyImportsAllowed)
	}

	return importsBelowRoot && !importsBelowBase && !importsAllowed
}

var rootPackage string

func main() {
	if len(os.Args) != 3 {
		log.Fatalf("Usage: %s <root> <restrictions.yaml>", os.Args[0])
	}

	rootPackage = os.Args[1]
	configFile := os.Args[2]
	importRestrictions, err := loadImportRestrictions(configFile)
	if err != nil {
		log.Fatalf("Failed to load import restrictions: %v", err)
	}

	foundForbiddenImports := false
	for _, restriction := range importRestrictions {
		baseDir := restriction.BaseDir
		if filepath.IsAbs(baseDir) {
			log.Fatalf("%q appears to be an absolute path", baseDir)
		}
		if !strings.HasPrefix(baseDir, "./") {
			baseDir = "./" + baseDir
		}
		baseDir = strings.TrimRight(baseDir, "/")
		log.Printf("Inspecting imports under %s/...\n", baseDir)

		packages, err := resolvePackageTree(baseDir)
		if err != nil {
			log.Fatalf("Failed to resolve package tree: %v", err)
		} else if len(packages) == 0 {
			log.Fatalf("Found no packages under tree %s", baseDir)
		}

		log.Printf("- validating imports for %d packages", len(packages))
		restrictionViolated := false
		for _, pkg := range packages {
			if forbidden, err := restriction.ForbiddenImportsFor(pkg); err != nil {
				log.Fatalf("-- failed to validate imports: %v", err)
			} else if len(forbidden) != 0 {
				logForbiddenPackages(pkg.ImportPath, forbidden)
				restrictionViolated = true
			}
		}
		if restrictionViolated {
			foundForbiddenImports = true
			log.Println("- FAIL")
		} else {
			log.Println("- OK")
		}
	}

	if foundForbiddenImports {
		os.Exit(1)
	}
}

func loadImportRestrictions(configFile string) ([]ImportRestriction, error) {
	config, err := os.ReadFile(configFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load configuration from %s: %v", configFile, err)
	}

	var importRestrictions []ImportRestriction
	if err := yaml.Unmarshal(config, &importRestrictions); err != nil {
		return nil, fmt.Errorf("failed to unmarshal from %s: %v", configFile, err)
	}

	return importRestrictions, nil
}

func resolvePackageTree(treeBase string) ([]Package, error) {
	cmd := "go"
	args := []string{"list", "-json", fmt.Sprintf("%s/...", treeBase)}
	c := exec.Command(cmd, args...)
	stdout, err := c.Output()
	if err != nil {
		var message string
		if ee, ok := err.(*exec.ExitError); ok {
			message = fmt.Sprintf("%v\n%v", ee, string(ee.Stderr))
		} else {
			message = fmt.Sprintf("%v", err)
		}
		return nil, fmt.Errorf("failed to run `%s %s`: %v", cmd, strings.Join(args, " "), message)
	}

	packages, err := decodePackages(bytes.NewReader(stdout))
	if err != nil {
		return nil, fmt.Errorf("failed to decode packages: %v", err)
	}

	return packages, nil
}

func decodePackages(r io.Reader) ([]Package, error) {
	// `go list -json` concatenates package definitions
	// instead of emitting a single valid JSON, so we
	// need to stream the output to decode it into the
	// data we are looking for instead of just using a
	// simple JSON decoder on stdout
	var packages []Package
	decoder := json.NewDecoder(r)
	for decoder.More() {
		var pkg Package
		if err := decoder.Decode(&pkg); err != nil {
			return nil, fmt.Errorf("invalid package: %v", err)
		}
		packages = append(packages, pkg)
	}

	return packages, nil
}

func logForbiddenPackages(base string, forbidden []string) {
	log.Printf("-- found forbidden imports for %s:\n", base)
	for _, forbiddenPackage := range forbidden {
		log.Printf("--- %s\n", forbiddenPackage)
	}
}
