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
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
)

const (
	Name      = "unversioned-helper-gen"
	TagName   = "+k8s:unversioned-gen"
	TagPrefix = TagName + "="
)

// Code generator for generating helper utilities for unversioned (internal) Kubernetes types from
// their versioned counterparts.
//
// Limitations:
//   - Currently, only translation from `k8s.io/api/core/v1` to `k8s.io/kubernetes/pkg/apis/core` is supported.
//
// Tag Usage:
//
//	Tag the input file with:
//	  // +k8s:unversioned-gen=k8s.io/kubernetes/PATH
//	where PATH is the package path for the unversioned helper (e.g., `pkg/apis/core`).
//	The generator will remove this tag from the generated output.
//
// Output:
//
//	The generated file will be written to the directory specified by the tag, relative to the Kubernetes root.
//	The output filename will be the input filename with "_generated" appended before the extension
//	(e.g., `foo.go` -> `foo_generated.go`, `foo_test.go` -> `foo_generated_test.go`).
//	On success, the tool prints the relative destination directory to stdout.
//
// CLI Usage:
//
//	unversioned-helper-gen -input-file <path-to-input-go-file>
func main() {
	var inputFile string
	flag.StringVar(&inputFile, "input-file", "", "input file path")
	flag.Parse()

	if inputFile == "" {
		log.Fatalf("input-file must be provided")
	}

	kubeRoot, err := findKubeRoot()
	if err != nil {
		log.Fatalf("Error finding kube root: %v", err)
	}

	g := &generator{kubeRoot: kubeRoot}
	outDir, err := g.processFile(inputFile)
	if err != nil {
		log.Fatalf("Error: %v", err)
	}
	fmt.Println(outDir) // Print destination directory to stdout
}

func (g *generator) processFile(input string) (string, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, input, nil, parser.ParseComments)
	if err != nil {
		return "", fmt.Errorf("failed to parse file %s: %v", input, err)
	}

	tag, err := extractAndRemoveTag(f)
	if err != nil {
		return "", fmt.Errorf("%s: invalid tag: %v", input, err)
	}

	relDstDir, err := validateTag(tag)
	if err != nil {
		return "", fmt.Errorf("%s: invalid tag %q: %v", input, tag, err)
	}

	outputDir := filepath.Join(g.kubeRoot, relDstDir)
	pkgName := filepath.Base(outputDir)
	outputImportPath, err := g.getOutputImportPath(outputDir)
	if err != nil {
		return "", err
	}

	if err := verifyOutputDir(outputDir); err != nil {
		return "", err
	}

	dstPath, err := determineOutputPath(input, outputDir)
	if err != nil {
		return "", err
	}

	err = g.translateAST(fset, f, input, dstPath, pkgName, outputImportPath)
	if err != nil {
		return "", fmt.Errorf("failed to translate %s: %v", input, err)
	}

	return relDstDir, nil
}

// verifyOutputDir checks if the output directory exists and is indeed a directory.
func verifyOutputDir(path string) error {
	fi, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("output directory %s does not exist", path)
		}
		return fmt.Errorf("failed to stat output directory %s: %v", path, err)
	}
	if !fi.IsDir() {
		return fmt.Errorf("output path %s is not a directory", path)
	}
	return nil
}

// determineOutputPath returns the output file path based on the input file name and output directory.
// It appends "_generated" to the base name, preserving "_test" suffix if present.
func determineOutputPath(input, outputDir string) (string, error) {
	filename := filepath.Base(input)
	if !strings.HasSuffix(filename, ".go") {
		return "", fmt.Errorf("input file %s is not a .go file", input)
	}
	name := strings.TrimSuffix(filename, ".go")
	if strings.HasSuffix(name, "_test") {
		baseName := strings.TrimSuffix(name, "_test")
		return filepath.Join(outputDir, baseName+"_generated_test.go"), nil
	}
	return filepath.Join(outputDir, name+"_generated.go"), nil
}

// extractAndRemoveTag finds the generator tag in the file comments,
// removes it from the AST, and returns its value.
func extractAndRemoveTag(f *ast.File) (string, error) {
	var tagValue string
	var tagFound bool

	var newComments []*ast.CommentGroup
	for _, group := range f.Comments {
		var newList []*ast.Comment
		for _, comment := range group.List {
			if trimmed, found := parseTag(comment.Text); found {
				if tagFound {
					return "", fmt.Errorf("multiple %s tags found", TagName)
				}
				tagValue = trimmed
				tagFound = true
				continue // Skip this comment (remove it)
			}
			newList = append(newList, comment)
		}
		if len(newList) > 0 {
			group.List = newList
			newComments = append(newComments, group)
		}
	}
	f.Comments = newComments

	if !tagFound {
		return "", fmt.Errorf("no %s tag found", TagName)
	}
	if tagValue == "" {
		return "", fmt.Errorf("tag %s cannot be empty", TagName)
	}
	return tagValue, nil
}

// parseTag checks if a comment line contains the generator tag and returns its value.
func parseTag(commentText string) (string, bool) {
	if !strings.HasPrefix(commentText, "//") {
		return "", false
	}
	trimmed := strings.TrimSpace(strings.TrimPrefix(commentText, "//"))
	if !strings.HasPrefix(trimmed, TagPrefix) {
		return "", false
	}
	return strings.TrimPrefix(trimmed, TagPrefix), true
}

// validateTag checks if the tag value is a valid destination path.
func validateTag(tag string) (string, error) {
	if !strings.HasPrefix(tag, "k8s.io/kubernetes/") {
		return "", fmt.Errorf("tag must start with k8s.io/kubernetes/")
	}
	relPath := strings.TrimPrefix(tag, "k8s.io/kubernetes/")
	if relPath == "" {
		return "", fmt.Errorf("destination path cannot be empty")
	}
	return relPath, nil
}

// translateAST modifies the AST to adapt it for the destination package.
// It changes the package name, updates imports, and rewrites type references.
func (g *generator) translateAST(fset *token.FileSet, f *ast.File, input, output, pkg, outputImport string) error {
	f.Name.Name = pkg

	selfAlias, err := updateImports(f, outputImport)
	if err != nil {
		return err
	}

	if outputImport != "" {
		astutil.DeleteImport(fset, f, outputImport)
	}

	rewriteReferences(f, selfAlias)

	return g.writeGeneratedFile(fset, f, input, output, pkg)
}

// updateImports updates the import paths in the AST.
// It translates versioned API imports (currently only v1 core) to their internal counterparts.
// It returns the alias used for the outputImport, if it was imported, which is used later
// to remove self-references.
func updateImports(f *ast.File, outputImport string) (string, error) {
	var selfAlias string
	for _, imp := range f.Imports {
		path := strings.Trim(imp.Path.Value, `"`)

		if outputImport != "" && path == outputImport {
			if imp.Name != nil {
				selfAlias = imp.Name.Name
			} else {
				// If imported without alias, we assume it refers to the package name
				// which matches the target package name we set.
				selfAlias = f.Name.Name
			}
			continue
		}

		// Translate the versioned types to internal types.
		// TODO: extend this to work for types outside of core.
		if strings.HasPrefix(path, `k8s.io/api/`) {
			if path == `k8s.io/api/core/v1` {
				imp.Path.Value = `"k8s.io/kubernetes/pkg/apis/core"`
				imp.Name = &ast.Ident{Name: "core"}
			} else {
				return "", fmt.Errorf("file imports unsupported api %s. Only k8s.io/api/core/v1 is supported for now.", path)
			}
		}
	}
	return selfAlias, nil
}

// rewriteReferences updates references in the AST.
// It replaces "v1" package qualifier with "core" (matching the translated import).
// It also removes the package qualifier for the package we are generating into (selfAlias)
// because those types are now local to the generated package.
func rewriteReferences(f *ast.File, selfAlias string) {
	astutil.Apply(f, func(c *astutil.Cursor) bool {
		switch n := c.Node().(type) {
		case *ast.SelectorExpr:
			if ident, ok := n.X.(*ast.Ident); ok {
				// TODO: generalize this for types outside of core.
				if ident.Name == "v1" {
					ident.Name = "core"
				} else if selfAlias != "" && ident.Name == selfAlias {
					// If the selector was the alias for the internal package, drop it.
					// e.g., if we imported the target package as "helper", and we had "helper.SomeType",
					// we rewrite it to just "SomeType" since we are now in that package.
					c.Replace(n.Sel)
				}
			}
		}
		return true
	}, nil)
}

// writeGeneratedFile formats the AST and writes it to the output file with a generated header.
func (g *generator) writeGeneratedFile(fset *token.FileSet, f *ast.File, input, output, pkg string) error {
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, f); err != nil {
		return fmt.Errorf("failed to format node: %v", err)
	}

	// Inject the header before the package declaration
	header := g.getGeneratedHeader(input)
	pkgDecl := "package " + pkg
	outStr := bytes.Replace(buf.Bytes(), []byte(pkgDecl), []byte(header+pkgDecl), 1)

	if err := os.WriteFile(output, outStr, 0644); err != nil {
		return fmt.Errorf("failed to write output %s: %v", output, err)
	}
	fmt.Fprintf(os.Stderr, "Generated %s from %s\n", output, input)
	return nil
}

// getOutputImportPath gets the import path that refers to the output package.
func (g *generator) getOutputImportPath(outputDir string) (string, error) {
	absOutput, err := filepath.Abs(outputDir)
	if err != nil {
		return "", fmt.Errorf("failed to get absolute path for %s: %v", outputDir, err)
	}
	rel, err := filepath.Rel(g.kubeRoot, absOutput)
	if err != nil {
		return "", fmt.Errorf("failed to get relative path: %v", err)
	}
	return "k8s.io/kubernetes/" + filepath.ToSlash(rel), nil
}

// getGeneratedHeader returns the header comment block for the generated file.
func (g *generator) getGeneratedHeader(inputPath string) string {
	relPath := g.getRelativeSourcePath(inputPath)
	return fmt.Sprintf("// Code generated by %s. DO NOT EDIT.\n// Source: %s\n\n", Name, relPath)
}

// getRelativeSourcePath returns the path of the input file relative to the kube root.
func (g *generator) getRelativeSourcePath(inputPath string) string {
	absInput, err := filepath.Abs(inputPath)
	if err != nil {
		return inputPath
	}
	relInput, err := filepath.Rel(g.kubeRoot, absInput)
	if err != nil {
		return inputPath
	}
	return relInput
}

type generator struct {
	kubeRoot string
}

// findKubeRoot returns the root of the kubernetes source tree.
// It first checks the KUBE_ROOT environment variable, and falls back to the current working directory.
func findKubeRoot() (string, error) {
	if kubeRoot := os.Getenv("KUBE_ROOT"); kubeRoot != "" {
		return kubeRoot, nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get current working directory: %v", err)
	}
	return cwd, nil
}
