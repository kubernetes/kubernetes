/*
Copyright 2024 The Kubernetes Authors.

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

package cmd

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/cobra"
)

var (
	newFeaturesFile string
	oldFeaturesFile string
	packagePrefix   string
)

// NewFeatureGatesCommand returns the cobra command for "feature-gates".
func NewFeatureGatesCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "feature-gates <subcommand>",
		Short: "Commands related to feature gate verifications",
	}

	cmd.AddCommand(NewNoNewUnversionedCommand())
	cmd.AddCommand(NewAlphabeticOrderCommand())
	return cmd
}

func NewNoNewUnversionedCommand() *cobra.Command {
	cmd := cobra.Command{
		Use:   "verify-no-new-unversioned",
		Short: "Verifies no new unversioned feature gates are added.",
		Run:   noNewUnversionedCmdFunc,
	}

	cmd.Flags().StringVar(&newFeaturesFile, "new-features-file", "pkg/features/kube_features.go", "relative path of the kube_features.go file to analyze")
	cmd.Flags().StringVar(&oldFeaturesFile, "old-features-file", "", "relative path of the master head kube_features.go file to compare the new kube_features with. If unspecified, will try to download from master branch on github.")

	return &cmd
}

func NewAlphabeticOrderCommand() *cobra.Command {
	cmd := cobra.Command{
		Use:   "verify-alphabetic-order",
		Short: "Verifies all features are added in alphabetic order.",
		Run:   alphabeticOrderCmdFunc,
	}

	cmd.Flags().StringVar(&newFeaturesFile, "features-file", "pkg/features/kube_features.go", "relative path of the kube_features.go file to analyze")
	cmd.Flags().StringVar(&packagePrefix, "package-prefix", "", "if specified, only include features from the imported package with the specified prefix. Otherwise all features should be alphabetically ordered.")
	return &cmd
}

func noNewUnversionedCmdFunc(cmd *cobra.Command, args []string) {
	if err := verifyNoNewUnversionedFeatureSpec(newFeaturesFile, oldFeaturesFile); err != nil {
		panic(err)
	}
}

func alphabeticOrderCmdFunc(cmd *cobra.Command, args []string) {
	fset := token.NewFileSet()
	if err := verifyAlphabeticOrderInFeatureSpecMap(fset, newFeaturesFile, packagePrefix, false); err != nil {
		panic(err)
	}
	if err := verifyAlphabeticOrderInFeatureSpecMap(fset, newFeaturesFile, packagePrefix, true); err != nil {
		panic(err)
	}
}

func verifyAlphabeticOrderInFeatureSpecMap(fset *token.FileSet, filePath, pkgPrefix string, versioned bool) error {
	features := extractFeatureSpecMapKeysFromFile(fset, filePath, versioned)
	unsortedFeatures := []string{}
	for _, f := range features {
		if !strings.HasPrefix(f, pkgPrefix) {
			continue
		}
		parts := strings.Split(f, ".")
		// features should be order by their feature names irrespective of upper or lower cases.
		// features from the same package should also be grouped together.
		if len(parts) < 2 {
			unsortedFeatures = append(unsortedFeatures, strings.ToUpper(f))
		} else {
			unsortedFeatures = append(unsortedFeatures, strings.Join([]string{parts[0], strings.ToUpper(parts[1])}, "."))
		}
	}
	if len(unsortedFeatures) < 2 {
		return nil
	}
	featuresSorted := make([]string, len(unsortedFeatures))
	copy(featuresSorted, unsortedFeatures)
	sort.Strings(featuresSorted)
	if diff := cmp.Diff(unsortedFeatures, featuresSorted); diff != "" {
		return fmt.Errorf("features in %s are not in alphabetic order, diff: %s", newFeaturesFile, diff)
	}
	return nil
}

// verifyNoNewUnversionedFeatureSpec compares the feature specs in the current features file
// with feature specs in the master branch, and verifies that no new features are added as unversioned feature specs.
func verifyNoNewUnversionedFeatureSpec(newFilePath, oldFilePath string) error {
	// Create a FileSet to work with
	fset := token.NewFileSet()
	if oldFilePath == "" {
		oldFilePath = filepath.Join("__masterbranch", newFilePath)
	}
	if _, err := os.Stat(oldFilePath); err != nil {
		headFileURL := "https://raw.githubusercontent.com/kubernetes/kubernetes/master/" + newFilePath
		if err := downloadFile(oldFilePath, headFileURL); err != nil {
			panic(err)
		}
	}
	featuresOld := extractFeatureSpecMapKeysFromFile(fset, oldFilePath, false)
	featuresNew := extractFeatureSpecMapKeysFromFile(fset, newFilePath, false)
	oldFeatureSet := make(map[string]struct{})
	newFeatures := []string{}
	for _, f := range featuresOld {
		oldFeatureSet[f] = struct{}{}
	}
	for _, f := range featuresNew {
		if _, found := oldFeatureSet[f]; !found {
			newFeatures = append(newFeatures, f)
		}
	}
	if len(newFeatures) > 0 {
		return fmt.Errorf("%s: new features added to FeatureSpec map! %v\nPlease add new features through VersionedSpecs map ONLY! ", newFilePath, newFeatures)
	}
	return nil
}

// extractFeatureSpecMapKeysFromFile extracts all the the keys from
// map[featuregate.Feature]featuregate.FeatureSpec or map[featuregate.Feature]featuregate.VersionedSpecs from the given file.
func extractFeatureSpecMapKeysFromFile(fset *token.FileSet, filePath string, versioned bool) (keys []string) {
	// Parse the file and create an AST
	absFilePath, err := filepath.Abs(filePath)
	if err != nil {
		panic(err)
	}
	file, err := parser.ParseFile(fset, absFilePath, nil, parser.AllErrors)
	if err != nil {
		panic(err)
	}
	aliasMap := importAliasMap(file.Imports)
	for _, d := range file.Decls {
		if gd, ok := d.(*ast.GenDecl); ok && (gd.Tok == token.CONST || gd.Tok == token.VAR) {
			for _, spec := range gd.Specs {
				if vspec, ok := spec.(*ast.ValueSpec); ok {
					for _, name := range vspec.Names {
						for _, value := range vspec.Values {
							mapKeys := extractFeatureSpecMapKeys(value, aliasMap, versioned)
							if len(mapKeys) > 0 {
								fmt.Printf("found FeatureSpecMap: %s\n", name)
								keys = append(keys, mapKeys...)
							}
						}
					}
				}
			}
		}
	}
	if versioned {
		fmt.Printf("Found %d versioned FeatureSpecMapKeys from file: %s\n", len(keys), filePath)
	} else {
		fmt.Printf("Found %d FeatureSpecMapKeys from file: %s\n", len(keys), filePath)
	}
	return
}

// extractFeatureSpecMapKeys extracts all the the keys from
// map[featuregate.Feature]featuregate.FeatureSpec or map[featuregate.Feature]featuregate.VersionedSpecs.
func extractFeatureSpecMapKeys(v ast.Expr, aliasMap map[string]string, versioned bool) (keys []string) {
	cl, ok := v.(*ast.CompositeLit)
	if !ok {
		return
	}
	mt, ok := cl.Type.(*ast.MapType)
	if !ok {
		return
	}
	if !isFeatureSpecType(mt.Value, aliasMap, versioned) {
		return
	}
	for _, elt := range cl.Elts {
		kv, ok := elt.(*ast.KeyValueExpr)
		if !ok {
			continue
		}
		keys = append(keys, identifierName(kv.Key))
	}
	return
}

func isFeatureSpecType(v ast.Expr, aliasMap map[string]string, versioned bool) bool {
	typeName := "FeatureSpec"
	if versioned {
		typeName = "VersionedSpecs"
	}
	pkg := "\"k8s.io/component-base/featuregate\""
	alias, ok := aliasMap[pkg]
	if ok {
		typeName = alias + "." + typeName
	}
	return identifierName(v) == typeName
}
