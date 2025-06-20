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

package sortedfeatures

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"testing"

	"golang.org/x/tools/go/analysis"
)

func TestIsFeatureGateDeclaration(t *testing.T) {
	tests := []struct {
		name     string
		src      string
		expected bool
	}{
		{
			name: "Feature type declaration",
			src: `
package test
import "k8s.io/component-base/featuregate"
const (
	FeatureA featuregate.Feature = "FeatureA"
)`,
			expected: true,
		},
		{
			name: "WithFeature pattern",
			src: `
package test
import "k8s.io/kubernetes/test/e2e/framework"
var (
	FeatureA = framework.WithFeature(framework.ValidFeatures.Add("FeatureA"))
)`,
			expected: true,
		},
		{
			name: "Feature pattern",
			src: `
package test
import "k8s.io/kubernetes/test/e2e/framework"
var (
	FeatureA = framework.Feature(framework.ValidFeatures.Add("FeatureA"))
)`,
			expected: true,
		},
		{
			name: "String literal matching name",
			src: `
package test
const (
	FeatureA = "FeatureA"
)`,
			expected: true,
		},
		{
			name: "Regular variable declaration",
			src: `
package test
const (
	RegularVar = 42
)`,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, "test.go", tt.src, 0)
			if err != nil {
				t.Fatalf("Failed to parse test file: %v", err)
			}

			var found bool
			ast.Inspect(f, func(n ast.Node) bool {
				if decl, ok := n.(*ast.GenDecl); ok && (decl.Tok == token.CONST || decl.Tok == token.VAR) {
					for _, spec := range decl.Specs {
						if valueSpec, ok := spec.(*ast.ValueSpec); ok {
							if isFeatureGateDeclaration(valueSpec) {
								found = true
							}
						}
					}
				}
				return true
			})

			if found != tt.expected {
				t.Errorf("isFeatureGateDeclaration() = %v, want %v", found, tt.expected)
			}
		})
	}
}

func TestNewAnalyzer(t *testing.T) {
	analyzer := NewAnalyzer()
	if analyzer.Name != "sortedfeatures" {
		t.Errorf("Expected analyzer name to be 'sortedfeatures', got '%s'", analyzer.Name)
	}
}

func TestNewAnalyzerWithConfig(t *testing.T) {
	config := Config{
		Debug: true,
		Files: []string{"test.go"},
	}
	analyzer := NewAnalyzerWithConfig(config)
	if analyzer.Name != "sortedfeatures" {
		t.Errorf("Expected analyzer name to be 'sortedfeatures', got '%s'", analyzer.Name)
	}
}

func TestRunWithEmptyFileList(t *testing.T) {
	// Test that the analyzer handles empty file list gracefully
	pass := &analysis.Pass{
		Fset:     token.NewFileSet(),
		Files:    []*ast.File{},
		ResultOf: make(map[*analysis.Analyzer]interface{}),
		Report:   func(d analysis.Diagnostic) { t.Errorf("Report should not be called") },
	}

	// Run the analyzer's run function directly
	result, err := run(pass, Config{})
	if err != nil {
		t.Errorf("run returned an error: %v", err)
	}
	if result != nil {
		t.Errorf("Expected nil result, got: %v", result)
	}
}

func TestRunWithDefaultFiles(t *testing.T) {
	// Test that the analyzer works with default files
	// Create a mock analysis.Pass
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "pkg/features/kube_features.go", `
package features

type Feature string

const (
	FeatureA Feature = "FeatureA"
	FeatureB Feature = "FeatureB"
)
`, 0)
	if err != nil {
		t.Fatalf("Failed to parse test file: %v", err)
	}

	pass := &analysis.Pass{
		Fset:     fset,
		Files:    []*ast.File{f},
		ResultOf: make(map[*analysis.Analyzer]interface{}),
		Report:   func(d analysis.Diagnostic) {},
	}

	// Run the analyzer's run function directly
	_, err = run(pass, Config{})
	if err != nil {
		t.Errorf("run returned an error: %v", err)
	}
}

func TestRunWithNonTargetFile(t *testing.T) {
	// Test that the analyzer ignores non-target files
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "some/random/file.go", `
package features

type Feature string

const (
	FeatureB Feature = "FeatureB"
	FeatureA Feature = "FeatureA"  // Unsorted, but should be ignored
)
`, 0)
	if err != nil {
		t.Fatalf("Failed to parse test file: %v", err)
	}

	var reportCalled bool
	pass := &analysis.Pass{
		Fset:     fset,
		Files:    []*ast.File{f},
		ResultOf: make(map[*analysis.Analyzer]interface{}),
		Report:   func(d analysis.Diagnostic) { reportCalled = true },
	}

	// Run the analyzer's run function directly
	_, err = run(pass, Config{})
	if err != nil {
		t.Errorf("run returned an error: %v", err)
	}

	// Check that Report was NOT called for non-target files
	if reportCalled {
		t.Errorf("Report should not be called for non-target files")
	}
}

func TestRunWithSpecifiedFiles(t *testing.T) {
	// Test that the analyzer works with specified files
	config := Config{
		Files: []string{"custom_file.go"},
	}

	// Create a mock analysis.Pass
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "custom_file.go", `
package features

type Feature string

const (
	FeatureA Feature = "FeatureA"
	FeatureB Feature = "FeatureB"
)
`, 0)
	if err != nil {
		t.Fatalf("Failed to parse test file: %v", err)
	}

	pass := &analysis.Pass{
		Fset:     fset,
		Files:    []*ast.File{f},
		ResultOf: make(map[*analysis.Analyzer]interface{}),
		Report:   func(d analysis.Diagnostic) {},
	}

	// Run the analyzer's run function directly
	_, err = run(pass, config)
	if err != nil {
		t.Errorf("run returned an error: %v", err)
	}
}

func TestCheckSorting(t *testing.T) {
	// Test the checkSorting function directly
	src := `
package test

type Feature string

const (
	// These are NOT properly sorted
	FeatureB Feature = "FeatureB"
	FeatureA Feature = "FeatureA"
	FeatureC Feature = "FeatureC"
)
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "test.go", src, 0)
	if err != nil {
		t.Fatalf("Failed to parse test file: %v", err)
	}

	var foundDecl *ast.GenDecl
	ast.Inspect(f, func(n ast.Node) bool {
		if decl, ok := n.(*ast.GenDecl); ok && decl.Tok == token.CONST {
			foundDecl = decl
			return false
		}
		return true
	})

	if foundDecl == nil {
		t.Fatalf("Failed to find const declaration")
	}

	// Create a mock analysis.Pass
	var reportCalled bool
	pass := &analysis.Pass{
		Fset:     fset,
		Files:    []*ast.File{f},
		ResultOf: make(map[*analysis.Analyzer]interface{}),
		Report:   func(d analysis.Diagnostic) { reportCalled = true },
	}

	// Call checkSorting
	checkSorting(pass, foundDecl)

	// Check that Report was called
	if !reportCalled {
		t.Errorf("Expected Report to be called for unsorted features")
	}
}

func TestSortedFeatures(t *testing.T) {
	// Test with properly sorted features
	src := `
package test

type Feature string

const (
	// These are properly sorted
	FeatureA Feature = "FeatureA"
	FeatureB Feature = "FeatureB"
	FeatureC Feature = "FeatureC"
)
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "test.go", src, 0)
	if err != nil {
		t.Fatalf("Failed to parse test file: %v", err)
	}

	var foundDecl *ast.GenDecl
	ast.Inspect(f, func(n ast.Node) bool {
		if decl, ok := n.(*ast.GenDecl); ok && decl.Tok == token.CONST {
			foundDecl = decl
			return false
		}
		return true
	})

	if foundDecl == nil {
		t.Fatalf("Failed to find const declaration")
	}

	// Create a mock analysis.Pass
	var reportCalled bool
	pass := &analysis.Pass{
		Fset:     fset,
		Files:    []*ast.File{f},
		ResultOf: make(map[*analysis.Analyzer]interface{}),
		Report:   func(d analysis.Diagnostic) { reportCalled = true },
	}

	// Call checkSorting
	checkSorting(pass, foundDecl)

	// Check that Report was NOT called
	if reportCalled {
		t.Errorf("Report should not be called for sorted features")
	}
}

func TestUnsortedFeatures(t *testing.T) {
	// Test with unsorted features
	src := `
package test

type Feature string

const (
	// These are NOT properly sorted
	FeatureB Feature = "FeatureB"
	FeatureA Feature = "FeatureA"
	FeatureC Feature = "FeatureC"
)
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "test.go", src, 0)
	if err != nil {
		t.Fatalf("Failed to parse test file: %v", err)
	}

	var foundDecl *ast.GenDecl
	ast.Inspect(f, func(n ast.Node) bool {
		if decl, ok := n.(*ast.GenDecl); ok && decl.Tok == token.CONST {
			foundDecl = decl
			return false
		}
		return true
	})

	if foundDecl == nil {
		t.Fatalf("Failed to find const declaration")
	}

	// Create a mock analysis.Pass
	var reportCalled bool
	pass := &analysis.Pass{
		Fset:     fset,
		Files:    []*ast.File{f},
		ResultOf: make(map[*analysis.Analyzer]interface{}),
		Report:   func(d analysis.Diagnostic) { reportCalled = true },
	}

	// Call checkSorting
	checkSorting(pass, foundDecl)

	// Check that Report was called
	if !reportCalled {
		t.Errorf("Expected Report to be called for unsorted features")
	}
}

func TestVarBlockFeatures(t *testing.T) {
	// Test with var block features (like in test/e2e/feature/feature.go)
	src := `
package test

import "k8s.io/kubernetes/test/e2e/framework"

var (
	// These are properly sorted
	FeatureA = framework.WithFeature(framework.ValidFeatures.Add("FeatureA"))
	FeatureB = framework.WithFeature(framework.ValidFeatures.Add("FeatureB"))
	FeatureC = framework.WithFeature(framework.ValidFeatures.Add("FeatureC"))
)
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "test.go", src, 0)
	if err != nil {
		t.Fatalf("Failed to parse test file: %v", err)
	}

	var foundDecl *ast.GenDecl
	ast.Inspect(f, func(n ast.Node) bool {
		if decl, ok := n.(*ast.GenDecl); ok && decl.Tok == token.VAR {
			foundDecl = decl
			return false
		}
		return true
	})

	if foundDecl == nil {
		t.Fatalf("Failed to find var declaration")
	}

	// Create a mock analysis.Pass
	var reportCalled bool
	pass := &analysis.Pass{
		Fset:     fset,
		Files:    []*ast.File{f},
		ResultOf: make(map[*analysis.Analyzer]interface{}),
		Report:   func(d analysis.Diagnostic) { reportCalled = true },
	}

	// Call checkSorting
	checkSorting(pass, foundDecl)

	// Check that Report was NOT called
	if reportCalled {
		t.Errorf("Report should not be called for sorted features")
	}
}

func TestUnsortedVarBlockFeatures(t *testing.T) {
	// Test with unsorted var block features
	src := `
package test

import "k8s.io/kubernetes/test/e2e/framework"

var (
	// These are NOT properly sorted
	FeatureB = framework.WithFeature(framework.ValidFeatures.Add("FeatureB"))
	FeatureA = framework.WithFeature(framework.ValidFeatures.Add("FeatureA"))
	FeatureC = framework.WithFeature(framework.ValidFeatures.Add("FeatureC"))
)
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "test.go", src, 0)
	if err != nil {
		t.Fatalf("Failed to parse test file: %v", err)
	}

	var foundDecl *ast.GenDecl
	ast.Inspect(f, func(n ast.Node) bool {
		if decl, ok := n.(*ast.GenDecl); ok && decl.Tok == token.VAR {
			foundDecl = decl
			return false
		}
		return true
	})

	if foundDecl == nil {
		t.Fatalf("Failed to find var declaration")
	}

	// Create a mock analysis.Pass
	var reportCalled bool
	pass := &analysis.Pass{
		Fset:     fset,
		Files:    []*ast.File{f},
		ResultOf: make(map[*analysis.Analyzer]interface{}),
		Report:   func(d analysis.Diagnostic) { reportCalled = true },
	}

	// Call checkSorting
	checkSorting(pass, foundDecl)

	// Check that Report was called
	if !reportCalled {
		t.Errorf("Expected Report to be called for unsorted features")
	}
}

func TestSingleItemBlock(t *testing.T) {
	// Test with a block containing only one item
	src := `
package test

type Feature string

const (
	// Single item, should be skipped
	FeatureA Feature = "FeatureA"
)
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "test.go", src, 0)
	if err != nil {
		t.Fatalf("Failed to parse test file: %v", err)
	}

	var foundDecl *ast.GenDecl
	ast.Inspect(f, func(n ast.Node) bool {
		if decl, ok := n.(*ast.GenDecl); ok && decl.Tok == token.CONST {
			foundDecl = decl
			return false
		}
		return true
	})

	if foundDecl == nil {
		t.Fatalf("Failed to find const declaration")
	}

	// Create a mock analysis.Pass
	var reportCalled bool
	pass := &analysis.Pass{
		Fset:     fset,
		Files:    []*ast.File{f},
		ResultOf: make(map[*analysis.Analyzer]interface{}),
		Report:   func(d analysis.Diagnostic) { reportCalled = true },
	}

	// Call checkSorting
	checkSorting(pass, foundDecl)

	// Check that Report was NOT called
	if reportCalled {
		t.Errorf("Report should not be called for single item blocks")
	}
}

func TestMixedDeclarations(t *testing.T) {
	// Test with a block containing both feature gates and regular variables
	src := `
package test

type Feature string

const (
	// Mixed declarations
	RegularVar = 42
	FeatureB Feature = "FeatureB"
	FeatureA Feature = "FeatureA"  // Unsorted features
)
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "test.go", src, 0)
	if err != nil {
		t.Fatalf("Failed to parse test file: %v", err)
	}

	var foundDecl *ast.GenDecl
	ast.Inspect(f, func(n ast.Node) bool {
		if decl, ok := n.(*ast.GenDecl); ok && decl.Tok == token.CONST {
			foundDecl = decl
			return false
		}
		return true
	})

	if foundDecl == nil {
		t.Fatalf("Failed to find const declaration")
	}

	// Create a mock analysis.Pass
	var reportCalled bool
	pass := &analysis.Pass{
		Fset:     fset,
		Files:    []*ast.File{f},
		ResultOf: make(map[*analysis.Analyzer]interface{}),
		Report:   func(d analysis.Diagnostic) { reportCalled = true },
	}

	// Call checkSorting
	checkSorting(pass, foundDecl)

	// Check that Report was called
	if !reportCalled {
		t.Errorf("Expected Report to be called for unsorted features")
	}
}

// TestAnalyzerRunSimulatingGolangciLint is a test that simulates how golangci-lint would run the analyzer
// by creating a mock analysis.Pass and calling the Run method directly. If you run this test from the root
// of the repository, it will check all default target files defined in the analyzer's config without needing
// to run golangci-lint itself. If run from anywhere else, it will skip the test as the files won't be found.
func TestAnalyzerRunSimulatingGolangciLint(t *testing.T) {
	for _, filename := range defaultTargetFiles {
		t.Run(filename, func(t *testing.T) {

			// Check if file exists
			if _, err := os.Stat(filename); os.IsNotExist(err) {
				t.Skip("File not found; skipping test")
			}

			// Create a file set and parse a simple source
			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, filename, nil, 0)
			if err != nil {
				t.Fatalf("Failed to parse test file: %v", err)
			}

			// Create a mock analysis.Pass similar to what golangci-lint would create
			var diagnostics []analysis.Diagnostic
			pass := &analysis.Pass{
				Fset:     fset,
				Files:    []*ast.File{f},
				ResultOf: make(map[*analysis.Analyzer]interface{}),
				Report: func(d analysis.Diagnostic) {
					diagnostics = append(diagnostics, d)
				},
			}

			// Get the analyzer
			analyzer := NewAnalyzerWithConfig(Config{Debug: true})

			// Call the Run method directly as golangci-lint would
			result, err := analyzer.Run(pass)

			// Check that no error was returned
			if err != nil {
				t.Errorf("Analyzer Run returned an error: %v", err)
			}

			// Check that result is nil (our analyzer doesn't return any result)
			if result != nil {
				t.Errorf("Expected nil result for %s, got: %v", filename, result)
			}

			// Check that no diagnostics were reported for properly sorted features
			if len(diagnostics) > 0 {
				t.Errorf("Expected no diagnostics for %s, got: %v", filename, diagnostics)
			}
		})
	}
}

func TestDebugLogging(t *testing.T) {
	// This test doesn't actually verify the debug output, but ensures the code path is covered
	config := Config{
		Debug: true,
	}

	// Create a mock analysis.Pass
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "pkg/features/kube_features.go", `
package features

type Feature string

const (
	FeatureA Feature = "FeatureA"
	FeatureB Feature = "FeatureB"
)
`, 0)
	if err != nil {
		t.Fatalf("Failed to parse test file: %v", err)
	}

	pass := &analysis.Pass{
		Fset:     fset,
		Files:    []*ast.File{f},
		ResultOf: make(map[*analysis.Analyzer]interface{}),
		Report:   func(d analysis.Diagnostic) {},
	}

	// Run the analyzer's run function directly with debug enabled
	_, err = run(pass, config)
	if err != nil {
		t.Errorf("run returned an error: %v", err)
	}
}
