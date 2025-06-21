/*
Copyright 2025 The Kubernetes Authors.

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

// Package sortedfeatures implements a linter that checks if feature gates are sorted alphabetically.
package sortedfeatures

import (
	"fmt"
	"go/ast"
	"go/token"
	"path/filepath"
	"sort"
	"strings"

	"github.com/davecgh/go-spew/spew"
	"github.com/pmezard/go-difflib/difflib"
	"golang.org/x/tools/go/analysis"
)

// List of default files to check for feature gate sorting
var defaultTargetFiles = []string{
	"pkg/features/kube_features.go",
	"staging/src/k8s.io/apiserver/pkg/features/kube_features.go",
	"staging/src/k8s.io/client-go/features/known_features.go",
	"staging/src/k8s.io/controller-manager/pkg/features/kube_features.go",
	"staging/src/k8s.io/apiextensions-apiserver/pkg/features/kube_features.go",
	"test/e2e/feature/feature.go",
	"test/e2e/environment/environment.go",
}

// Config holds the configuration for the sortedfeatures analyzer
type Config struct {
	// Files contains files to check. If specified, only these files will be checked.
	Files []string
	// Debug enables debug logging
	Debug bool
}

// NewAnalyzer returns a new sortedfeatures analyzer.
func NewAnalyzer() *analysis.Analyzer {
	return NewAnalyzerWithConfig(Config{})
}

// NewAnalyzerWithConfig returns a new sortedfeatures analyzer with the given configuration.
func NewAnalyzerWithConfig(config Config) *analysis.Analyzer {
	return &analysis.Analyzer{
		Name: "sortedfeatures",
		Doc:  "Checks if feature gates are sorted alphabetically in const and var blocks",
		Run: func(pass *analysis.Pass) (interface{}, error) {
			return run(pass, config)
		},
	}
}

func run(pass *analysis.Pass, config Config) (interface{}, error) {
	// Check if there are any files to analyze
	if len(pass.Files) == 0 {
		// No files to analyze, return early
		return nil, nil
	}

	// Check if the current file is one of our target files
	filename := pass.Fset.File(pass.Files[0].Pos()).Name()
	isTargetFile := false

	// Determine which files to check
	var targetFiles []string
	if len(config.Files) > 0 {
		// If specific files are provided, only check those
		targetFiles = config.Files
	} else {
		// Otherwise use the default target files
		targetFiles = defaultTargetFiles
	}

	if config.Debug {
		fmt.Printf("Checking file: %s\n", filename)
	}

	for _, target := range targetFiles {
		if strings.HasSuffix(filename, target) || strings.HasSuffix(filename, filepath.Base(target)) {
			isTargetFile = true
			break
		}
	}

	if !isTargetFile {
		return nil, nil
	}

	for _, file := range pass.Files {
		ast.Inspect(file, func(n ast.Node) bool {
			switch decl := n.(type) {
			case *ast.GenDecl:
				if decl.Tok == token.CONST || decl.Tok == token.VAR {
					checkSorting(pass, decl)
				}
			}
			return true
		})
	}
	return nil, nil
}

// isFeatureGateDeclaration checks if a value spec is likely a feature gate declaration
func isFeatureGateDeclaration(spec *ast.ValueSpec) bool {
	// Check if there's a type with a selector expression that ends with Feature
	if spec.Type != nil {
		if selectorExpr, ok := spec.Type.(*ast.SelectorExpr); ok {
			return selectorExpr.Sel.Name == "Feature"
		}
	}

	// Check if there's a value that's a string literal matching the name pattern
	if len(spec.Values) > 0 {
		if basicLit, ok := spec.Values[0].(*ast.BasicLit); ok && basicLit.Kind == token.STRING {
			// Feature gates often have their name as the string value
			for _, name := range spec.Names {
				if strings.Contains(basicLit.Value, name.Name) {
					return true
				}
			}
		}

		// Check for framework.WithFeature pattern used in test/e2e/feature/feature.go
		if callExpr, ok := spec.Values[0].(*ast.CallExpr); ok {
			if selExpr, ok := callExpr.Fun.(*ast.SelectorExpr); ok {
				if selExpr.Sel.Name == "WithFeature" || selExpr.Sel.Name == "Feature" {
					return true
				}
			}
		}
	}

	return false
}

func checkSorting(pass *analysis.Pass, decl *ast.GenDecl) {
	if len(decl.Specs) <= 1 {
		return // Nothing to sort with just one item
	}

	// Extract feature names and their positions
	type featureInfo struct {
		Name string
	}

	var features []featureInfo
	var hasFeatureGates bool

	for _, spec := range decl.Specs {
		if valueSpec, ok := spec.(*ast.ValueSpec); ok {
			// Only check specs that look like feature gates
			if isFeatureGateDeclaration(valueSpec) {
				hasFeatureGates = true
				for _, name := range valueSpec.Names {
					features = append(features, featureInfo{
						Name: name.Name,
					})
				}
			}
		}
	}

	// If we didn't find any feature gates, don't check sorting
	if !hasFeatureGates || len(features) <= 1 {
		return
	}

	// Check if the features are sorted
	isSorted := true
	for i := 1; i < len(features); i++ {
		if strings.Compare(features[i-1].Name, features[i].Name) > 0 {
			isSorted = false
			break
		}
	}

	if !isSorted {
		// Create a sorted copy to show what the correct order should be
		sortedFeatures := make([]featureInfo, len(features))
		copy(sortedFeatures, features)
		sort.Slice(sortedFeatures, func(i, j int) bool {
			return strings.Compare(sortedFeatures[i].Name, sortedFeatures[j].Name) < 0
		})

		// Configure spew for better output
		spewConfig := spew.ConfigState{
			Indent:                  "  ",
			DisablePointerAddresses: true,
			DisableCapacities:       true,
			SortKeys:                true,
		}

		// Generate dumps of both current and expected orders
		currentDump := spewConfig.Sdump(features)
		expectedDump := spewConfig.Sdump(sortedFeatures)

		// Create a unified diff between the two dumps
		diff := difflib.UnifiedDiff{
			A:        difflib.SplitLines(currentDump),
			B:        difflib.SplitLines(expectedDump),
			FromFile: "Current",
			ToFile:   "Expected",
			Context:  3,
		}

		diffText, err := difflib.GetUnifiedDiffString(diff)
		if err != nil {
			pass.Reportf(decl.Pos(), "feature gates are not sorted alphabetically (error creating diff: %v)", err)
			return
		}

		// Report the issue with the diff
		pass.Reportf(decl.Pos(), "feature gates are not sorted alphabetically:\n%s", diffText)
	}
}
