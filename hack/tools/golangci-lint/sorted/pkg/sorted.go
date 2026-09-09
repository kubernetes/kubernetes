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

// Package pkg implements a linter that checks if feature gates are sorted alphabetically.
package pkg

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/token"
	"sort"
	"strings"

	"github.com/pmezard/go-difflib/difflib"
	"golang.org/x/tools/go/analysis"
)

// Config holds the configuration for the sorted analyzer
type Config struct {
	// Files contains files to check. If specified, only these files will be checked.
	Files []string
	// Debug enables debug logging
	Debug bool
}

// NewAnalyzer returns a new sorted analyzer.
func NewAnalyzer() *analysis.Analyzer {
	return NewAnalyzerWithConfig(Config{Debug: true})
}

// NewAnalyzerWithConfig returns a new sorted analyzer with the given configuration.
func NewAnalyzerWithConfig(config Config) *analysis.Analyzer {
	return &analysis.Analyzer{
		Name: "sorted",
		Doc:  "Checks if feature gates are sorted alphabetically in const and var blocks",
		Run: func(pass *analysis.Pass) (interface{}, error) {
			if config.Debug {
				fmt.Printf("Processing...\n")
			}
			return run(pass, config)
		},
	}
}

func isTargetFile(configFiles []string, filename string) bool {
	for _, item := range configFiles {
		if strings.HasSuffix(filename, item) {
			return true
		}
	}
	return false
}

func run(pass *analysis.Pass, config Config) (interface{}, error) {
	for _, file := range pass.Files {
		filename := pass.Fset.File(file.Pos()).Name()

		if !isTargetFile(config.Files, filename) {
			continue
		}

		// Check all declarations in the file
		for _, decl := range file.Decls {
			genDecl, ok := decl.(*ast.GenDecl)
			if !ok {
				continue
			}

			// Process var and const blocks for regular feature gates
			if (genDecl.Tok == token.VAR || genDecl.Tok == token.CONST) && len(genDecl.Specs) > 1 {
				// Extract features with their comments
				features := extractFeatures(genDecl, file.Comments)

				// Skip if no features were found
				if len(features) > 1 {
					// Sort features
					sortedFeatures := sortFeatures(features)

					// Check if the order has changed
					orderChanged := hasOrderChanged(features, sortedFeatures)

					if orderChanged {
						// Generate a diff to show what's wrong
						reportSortingIssue(pass, genDecl, features, sortedFeatures)
					}
				}
			}

			// Check for maps with feature gates as keys
			if genDecl.Tok == token.VAR {
				checkFeatureGateMaps(pass, genDecl)
			}
		}
	}
	return nil, nil
}

// checkFeatureGateMaps checks if maps with feature gates as keys have their keys sorted alphabetically
func checkFeatureGateMaps(pass *analysis.Pass, genDecl *ast.GenDecl) {
	for _, spec := range genDecl.Specs {
		valueSpec, ok := spec.(*ast.ValueSpec)
		if !ok || len(valueSpec.Names) == 0 || len(valueSpec.Values) == 0 {
			continue
		}

		// Check each value to see if it's a map with feature gates
		for _, value := range valueSpec.Values {
			compositeLit, ok := value.(*ast.CompositeLit)
			if !ok {
				continue
			}

			// Check if this is a map type
			mapType, ok := compositeLit.Type.(*ast.MapType)
			if !ok {
				continue
			}

			// Check if the key type is featuregate.Feature or contains "Feature"
			isFeatureGateMap := false

			// Check for SelectorExpr (e.g., featuregate.Feature)
			if selectorExpr, ok := mapType.Key.(*ast.SelectorExpr); ok {
				if selectorExpr.Sel.Name == "Feature" {
					isFeatureGateMap = true
				}
			}

			// Check for Ident (e.g., Feature)
			if ident, ok := mapType.Key.(*ast.Ident); ok {
				if ident.Name == "Feature" {
					isFeatureGateMap = true
				}
			}

			if !isFeatureGateMap {
				continue
			}

			// This is a map with feature gates as keys
			var features []Feature
			for _, elt := range compositeLit.Elts {
				keyValueExpr, ok := elt.(*ast.KeyValueExpr)
				if !ok {
					continue
				}

				// Get the key, which should be a feature gate identifier
				var featureName string

				// Handle different types of keys
				switch key := keyValueExpr.Key.(type) {
				case *ast.Ident:
					featureName = key.Name
				case *ast.SelectorExpr:
					// For selector expressions like genericfeatures.APIServerIdentity
					if x, ok := key.X.(*ast.Ident); ok {
						featureName = x.Name + "." + key.Sel.Name
					} else {
						continue
					}
				default:
					continue
				}

				features = append(features, Feature{
					Name:     featureName,
					Comments: []string{}, // No comments for map keys
				})
			}

			if len(features) <= 1 {
				continue
			}

			// Sort features
			sortedFeatures := sortFeatures(features)

			// Check if the order has changed
			orderChanged := hasOrderChanged(features, sortedFeatures)

			if orderChanged {
				// Generate a diff to show what's wrong
				reportMapSortingIssue(pass, genDecl, valueSpec.Names[0].Name, features, sortedFeatures)
			}
		}
	}
}

// Feature represents a feature declaration with its associated comments
type Feature struct {
	Name     string   // Name of the feature
	Comments []string // Comments associated with the feature
}

// extractFeatures extracts features from a GenDecl
func extractFeatures(decl *ast.GenDecl, comments []*ast.CommentGroup) []Feature {
	var features []Feature

	for _, spec := range decl.Specs {
		valueSpec, ok := spec.(*ast.ValueSpec)
		if !ok || len(valueSpec.Names) == 0 {
			continue
		}

		// Get the name of the feature
		name := valueSpec.Names[0].Name

		// Get comments for this feature
		var featureComments []string

		// Check for doc comments directly on the value spec
		if valueSpec.Doc != nil {
			for _, comment := range valueSpec.Doc.List {
				featureComments = append(featureComments, comment.Text)
			}
		} else {
			// Look for comments before this spec
			for _, cg := range comments {
				if cg.End()+1 == valueSpec.Pos() {
					for _, comment := range cg.List {
						featureComments = append(featureComments, comment.Text)
					}
				}
			}
		}

		features = append(features, Feature{
			Name:     name,
			Comments: featureComments,
		})
	}

	return features
}

// sortFeatures sorts features alphabetically by name
func sortFeatures(features []Feature) []Feature {
	sorted := make([]Feature, len(features))
	copy(sorted, features)

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Name < sorted[j].Name
	})

	return sorted
}

// hasOrderChanged checks if the order of features has changed
func hasOrderChanged(original, sorted []Feature) bool {
	if len(original) != len(sorted) {
		return true
	}

	for i := range original {
		if original[i].Name != sorted[i].Name {
			return true
		}
	}

	return false
}

// reportSortingIssue reports a linting issue with a diff showing the correct order
func reportSortingIssue(pass *analysis.Pass, decl *ast.GenDecl, current, sorted []Feature) {
	// Generate the original source code
	originalSource := generateSourceCode(decl.Tok, current)

	// Generate the sorted source code
	sortedSource := generateSourceCode(decl.Tok, sorted)

	// Create a unified diff between the original and sorted source
	diff := difflib.UnifiedDiff{
		A:        difflib.SplitLines(originalSource),
		B:        difflib.SplitLines(sortedSource),
		FromFile: "Current",
		ToFile:   "Expected",
		Context:  3,
	}

	diffText, err := difflib.GetUnifiedDiffString(diff)
	if err != nil {
		pass.Reportf(decl.Pos(), "not sorted alphabetically (error creating diff: %v)", err)
		return
	}

	// Report the issue with the diff
	pass.Reportf(decl.Pos(), "not sorted alphabetically (-got, +want):\n%s\n", stripHeader(diffText, 3))
}

func stripHeader(input string, n int) string {
	scanner := bufio.NewScanner(strings.NewReader(input))
	var result strings.Builder
	lineCount := 0

	for scanner.Scan() {
		lineCount++
		if lineCount > n {
			result.WriteString(scanner.Text() + "\n")
		}
	}

	return strings.TrimSuffix(result.String(), "\n")
}

// reportMapSortingIssue reports a linting issue for unsorted map keys
func reportMapSortingIssue(pass *analysis.Pass, decl *ast.GenDecl, mapName string, current, sorted []Feature) {
	// Generate the original source code
	originalSource := generateMapSourceCode(current)

	// Generate the sorted source code
	sortedSource := generateMapSourceCode(sorted)

	// Create a unified diff between the original and sorted source
	diff := difflib.UnifiedDiff{
		A:        difflib.SplitLines(originalSource),
		B:        difflib.SplitLines(sortedSource),
		FromFile: "Current Map Keys",
		ToFile:   "Expected Map Keys",
		Context:  3,
	}

	diffText, err := difflib.GetUnifiedDiffString(diff)
	if err != nil {
		pass.Reportf(decl.Pos(), "map '%s' keys not sorted alphabetically (error creating diff: %v)", mapName, err)
		return
	}

	// Report the issue with the diff
	pass.Reportf(decl.Pos(), "map '%s' keys not sorted alphabetically (-got, +want):\n%s\n", mapName, stripHeader(diffText, 3))
}

// generateMapSourceCode recreates the source code for map keys
func generateMapSourceCode(features []Feature) string {
	var sb strings.Builder

	sb.WriteString("map[featuregate.Feature]featuregate.VersionedSpecs{\n")

	// Add each feature key
	for _, feature := range features {
		sb.WriteString("\t")
		sb.WriteString(feature.Name)
		sb.WriteString(": ...,\n")
	}

	sb.WriteString("}")

	return sb.String()
}

// generateSourceCode recreates the source code from features
func generateSourceCode(tokenType token.Token, features []Feature) string {
	var sb strings.Builder

	// Start the block with the token type (var or const)
	sb.WriteString(tokenType.String())
	sb.WriteString(" (\n")

	// Add each feature with its comments
	for _, feature := range features {
		// Add comments
		for _, comment := range feature.Comments {
			sb.WriteString("\t")
			sb.WriteString(comment)
			sb.WriteString("\n")
		}

		// Add the feature declaration
		sb.WriteString("\t")
		sb.WriteString(feature.Name)
		sb.WriteString(" = ")
		// Since we don't have the actual value, we'll use a placeholder
		sb.WriteString("value")
		sb.WriteString("\n\n")
	}

	// Close the block
	sb.WriteString(")")

	return sb.String()
}
