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

package pkg

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"

	"github.com/pmezard/go-difflib/difflib"
	"golang.org/x/tools/go/analysis"
)

func TestNewAnalyzer(t *testing.T) {
	analyzer := NewAnalyzer()
	if analyzer.Name != "sorted" {
		t.Errorf("Expected analyzer name to be 'sorted', got '%s'", analyzer.Name)
	}
}

func TestNewAnalyzerWithConfig(t *testing.T) {
	config := Config{
		Debug: true,
		Files: []string{"test.go"},
	}
	analyzer := NewAnalyzerWithConfig(config)
	if analyzer.Name != "sorted" {
		t.Errorf("Expected analyzer name to be 'sorted', got '%s'", analyzer.Name)
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
`, parser.ParseComments)
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
`, parser.ParseComments)
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
	// Create a temporary file for testing
	tmpFile, err := os.CreateTemp("", "custom_file_*.go")
	if err != nil {
		t.Fatalf("Failed to create temporary file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	// Write test content to the file
	testContent := `
package features

type Feature string

const (
	FeatureA Feature = "FeatureA"
	FeatureB Feature = "FeatureB"
)
`
	if _, err := tmpFile.Write([]byte(testContent)); err != nil {
		t.Fatalf("Failed to write to temporary file: %v", err)
	}
	if err := tmpFile.Close(); err != nil {
		t.Fatalf("Failed to close temporary file: %v", err)
	}

	// Test that the analyzer works with specified files
	config := Config{
		Files: []string{tmpFile.Name()},
	}

	// Create a mock analysis.Pass
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, tmpFile.Name(), testContent, parser.ParseComments)
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

func TestExtractFeatures(t *testing.T) {
	src := `
package test

type Feature string

const (
	// Comment for FeatureA
	FeatureA Feature = "FeatureA"
	
	// Comment for FeatureB
	FeatureB Feature = "FeatureB"
)
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "test.go", src, parser.ParseComments)
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

	features := extractFeatures(foundDecl, f.Comments)

	if len(features) != 2 {
		t.Errorf("Expected 2 features, got %d", len(features))
	}

	if features[0].Name != "FeatureA" {
		t.Errorf("Expected first feature to be FeatureA, got %s", features[0].Name)
	}

	if features[1].Name != "FeatureB" {
		t.Errorf("Expected second feature to be FeatureB, got %s", features[1].Name)
	}

	if len(features[0].Comments) == 0 {
		t.Errorf("Expected comments for FeatureA, got none")
	}
}

func TestSortFeatures(t *testing.T) {
	features := []Feature{
		{Name: "FeatureB"},
		{Name: "FeatureA"},
		{Name: "FeatureC"},
	}

	sorted := sortFeatures(features)

	if sorted[0].Name != "FeatureA" {
		t.Errorf("Expected first feature to be FeatureA, got %s", sorted[0].Name)
	}

	if sorted[1].Name != "FeatureB" {
		t.Errorf("Expected second feature to be FeatureB, got %s", sorted[1].Name)
	}

	if sorted[2].Name != "FeatureC" {
		t.Errorf("Expected third feature to be FeatureC, got %s", sorted[2].Name)
	}
}

func TestHasOrderChanged(t *testing.T) {
	original := []Feature{
		{Name: "FeatureB"},
		{Name: "FeatureA"},
		{Name: "FeatureC"},
	}

	sorted := []Feature{
		{Name: "FeatureA"},
		{Name: "FeatureB"},
		{Name: "FeatureC"},
	}

	if !hasOrderChanged(original, sorted) {
		t.Errorf("Expected hasOrderChanged to return true for different orders")
	}

	if hasOrderChanged(sorted, sorted) {
		t.Errorf("Expected hasOrderChanged to return false for same order")
	}
}

func TestReportSortingIssue(t *testing.T) {
	current := []Feature{
		{Name: "FeatureB", Comments: []string{"// Comment for FeatureB"}},
		{Name: "FeatureA", Comments: []string{"// Comment for FeatureA"}},
		{Name: "FeatureC", Comments: []string{"// Comment for FeatureC"}},
	}

	sorted := []Feature{
		{Name: "FeatureA", Comments: []string{"// Comment for FeatureA"}},
		{Name: "FeatureB", Comments: []string{"// Comment for FeatureB"}},
		{Name: "FeatureC", Comments: []string{"// Comment for FeatureC"}},
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "test.go", `
package test

const (
	FeatureB = "FeatureB"
	FeatureA = "FeatureA"
	FeatureC = "FeatureC"
)
`, parser.ParseComments)
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

	var reportMessage string
	pass := &analysis.Pass{
		Fset:     fset,
		Files:    []*ast.File{f},
		ResultOf: make(map[*analysis.Analyzer]interface{}),
		Report:   func(d analysis.Diagnostic) { reportMessage = d.Message },
	}

	reportSortingIssue(pass, foundDecl, current, sorted)

	if reportMessage == "" {
		t.Errorf("Expected Report to be called")
	}

	// Check that the diff shows the features in the correct order
	if !strings.Contains(reportMessage, "FeatureB") || !strings.Contains(reportMessage, "FeatureA") || !strings.Contains(reportMessage, "FeatureC") {
		t.Errorf("Expected diff to contain all feature names, got: %s", reportMessage)
	}

	// Check that the diff contains the expected content
	expectedContent := "const ("
	if !strings.Contains(reportMessage, expectedContent) {
		t.Errorf("Expected diff to contain %q, got: %s", expectedContent, reportMessage)
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
	f, err := parser.ParseFile(fset, "test.go", src, parser.ParseComments)
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

	// Extract features
	features := extractFeatures(foundDecl, f.Comments)

	// Sort features
	sortedFeatures := sortFeatures(features)

	// Check if order changed
	if hasOrderChanged(features, sortedFeatures) {
		t.Errorf("Expected features to be already sorted")
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
	f, err := parser.ParseFile(fset, "test.go", src, parser.ParseComments)
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

	// Extract features
	features := extractFeatures(foundDecl, f.Comments)

	// Sort features
	sortedFeatures := sortFeatures(features)

	// Check if order changed
	if !hasOrderChanged(features, sortedFeatures) {
		t.Errorf("Expected features to be unsorted")
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
	f, err := parser.ParseFile(fset, "test.go", src, parser.ParseComments)
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

	// Extract features
	features := extractFeatures(foundDecl, f.Comments)

	// Sort features
	sortedFeatures := sortFeatures(features)

	// Check if order changed
	if hasOrderChanged(features, sortedFeatures) {
		t.Errorf("Expected features to be already sorted")
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
	f, err := parser.ParseFile(fset, "test.go", src, parser.ParseComments)
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

	// Extract features
	features := extractFeatures(foundDecl, f.Comments)

	if len(features) != 1 {
		t.Errorf("Expected 1 feature, got %d", len(features))
	}
}

func TestNonParenthesizedDeclarationsNotProcessed(t *testing.T) {
	// Test with non-parenthesized declarations
	src := `
package test

type Feature string

// First feature
const FeatureA Feature = "FeatureA"

// Second feature
const FeatureB Feature = "FeatureB"

// Third feature
const FeatureC Feature = "FeatureC"
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "test.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("Failed to parse test file: %v", err)
	}

	// Create a mock analysis.Pass
	var reportCalled bool
	pass := &analysis.Pass{
		Fset:     fset,
		Files:    []*ast.File{f},
		ResultOf: make(map[*analysis.Analyzer]interface{}),
		Report:   func(d analysis.Diagnostic) { reportCalled = true },
	}

	// Configure the analyzer to treat test.go as a target file
	config := Config{
		Files: []string{"test.go"},
	}

	// Run the analyzer
	_, err = run(pass, config)
	if err != nil {
		t.Errorf("run returned an error: %v", err)
	}

	// Check that Report was NOT called, since non-parenthesized declarations are skipped
	if reportCalled {
		t.Errorf("Expected Report not to be called for non-parenthesized declarations")
	}
}

// TestAnalyzerRunSimulatingGolangciLint is a test that simulates how golangci-lint would run the analyzer
// by creating a mock analysis.Pass and calling the Run method directly. If you run this test from the root
// of the repository, it will check all default target files defined in the analyzer's config without needing
// to run golangci-lint itself. If run from anywhere else, it will skip the test as the files won't be found.
func TestAnalyzerRunSimulatingGolangciLint(t *testing.T) {
	defaultTargetFiles := []string{
		"pkg/features/kube_features.go",
		"staging/src/k8s.io/apiserver/pkg/features/kube_features.go",
		"staging/src/k8s.io/client-go/features/known_features.go",
		"staging/src/k8s.io/controller-manager/pkg/features/kube_features.go",
		"staging/src/k8s.io/apiextensions-apiserver/pkg/features/kube_features.go",
		"test/e2e/feature/feature.go",
		"test/e2e/environment/environment.go",
	}
	for _, filename := range defaultTargetFiles {
		t.Run(filename, func(t *testing.T) {

			// Check if file exists
			if _, err := os.Stat(filename); os.IsNotExist(err) {
				t.Skip("File not found; skipping test")
			}

			// Create a file set and parse a simple source
			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, filename, nil, parser.ParseComments)
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
			analyzer := NewAnalyzerWithConfig(Config{Debug: true, Files: defaultTargetFiles})

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
`, parser.ParseComments)
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

func TestGenerateSourceCode(t *testing.T) {
	features := []Feature{
		{Name: "FeatureB", Comments: []string{"// Comment for FeatureB"}},
		{Name: "FeatureA", Comments: []string{"// Comment for FeatureA"}},
	}

	// Test const block generation
	constSource := generateSourceCode(token.CONST, features)
	expectedConstSource := `const (
	// Comment for FeatureB
	FeatureB = value

	// Comment for FeatureA
	FeatureA = value

)`

	if constSource != expectedConstSource {
		t.Errorf("Expected const source:\n%s\n\nGot:\n%s", expectedConstSource, constSource)
	}

	// Test var block generation
	varSource := generateSourceCode(token.VAR, features)
	expectedVarSource := `var (
	// Comment for FeatureB
	FeatureB = value

	// Comment for FeatureA
	FeatureA = value

)`

	if varSource != expectedVarSource {
		t.Errorf("Expected var source:\n%s\n\nGot:\n%s", expectedVarSource, varSource)
	}
}

func TestDiffGeneration(t *testing.T) {
	// Create unsorted features
	unsorted := []Feature{
		{Name: "FeatureC", Comments: []string{"// Comment for FeatureC"}},
		{Name: "FeatureA", Comments: []string{"// Comment for FeatureA"}},
		{Name: "FeatureB", Comments: []string{"// Comment for FeatureB"}},
	}

	// Create sorted features
	sorted := []Feature{
		{Name: "FeatureA", Comments: []string{"// Comment for FeatureA"}},
		{Name: "FeatureB", Comments: []string{"// Comment for FeatureB"}},
		{Name: "FeatureC", Comments: []string{"// Comment for FeatureC"}},
	}

	// Generate source code for both
	originalSource := generateSourceCode(token.CONST, unsorted)
	sortedSource := generateSourceCode(token.CONST, sorted)

	// Create a unified diff
	diff := difflib.UnifiedDiff{
		A:        difflib.SplitLines(originalSource),
		B:        difflib.SplitLines(sortedSource),
		FromFile: "Current",
		ToFile:   "Expected",
		Context:  3,
	}

	diffText, err := difflib.GetUnifiedDiffString(diff)
	if err != nil {
		t.Fatalf("Failed to generate diff: %v", err)
	}

	// Strip header to match the actual implementation
	diffText = stripHeader(diffText, 3)

	// Check that the diff contains expected content
	expectedLines := []string{
		"-\t// Comment for FeatureC",
		"-\tFeatureC = value",
	}

	for _, line := range expectedLines {
		if !strings.Contains(diffText, line) {
			t.Errorf("Expected diff to contain line: %q, but it was not found in:\n%s", line, diffText)
		}
	}

	// Check that the diff shows the correct order change
	if !strings.Contains(diffText, "FeatureA") || !strings.Contains(diffText, "FeatureB") || !strings.Contains(diffText, "FeatureC") {
		t.Errorf("Expected diff to contain all feature names, got: %s", diffText)
	}
}
