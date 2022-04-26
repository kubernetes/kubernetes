// Copyright 2020 Frederik Zipp. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gocyclo

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Analyze calculates the cyclomatic complexities of the functions and methods
// in the Go source code files in the given paths. If a path is a directory
// all Go files under that directory are analyzed recursively.
// Files with paths matching the 'ignore' regular expressions are skipped.
// The 'ignore' parameter can be nil, meaning that no files are skipped.
func Analyze(paths []string, ignore *regexp.Regexp) Stats {
	var stats Stats
	for _, path := range paths {
		info, err := os.Stat(path)
		if err != nil {
			log.Printf("could not get file info for path %q: %s\n", path, err)
			continue
		}
		if info.IsDir() {
			stats = analyzeDir(path, ignore, stats)
		} else {
			stats = analyzeFile(path, ignore, stats)
		}
	}
	return stats
}

func analyzeDir(dirname string, ignore *regexp.Regexp, stats Stats) Stats {
	filepath.WalkDir(dirname, func(path string, entry fs.DirEntry, err error) error {
		if err == nil && isGoFile(entry) {
			stats = analyzeFile(path, ignore, stats)
		}
		return err
	})
	return stats
}

func isGoFile(entry fs.DirEntry) bool {
	return !entry.IsDir() && strings.HasSuffix(entry.Name(), ".go")
}

func analyzeFile(path string, ignore *regexp.Regexp, stats Stats) Stats {
	if isIgnored(path, ignore) {
		return stats
	}
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
	if err != nil {
		log.Fatal(err)
	}
	return AnalyzeASTFile(f, fset, stats)
}

func isIgnored(path string, ignore *regexp.Regexp) bool {
	return ignore != nil && ignore.MatchString(path)
}

// AnalyzeASTFile calculates the cyclomatic complexities of the functions
// and methods in the abstract syntax tree (AST) of a parsed Go file and
// appends the results to the given Stats slice.
func AnalyzeASTFile(f *ast.File, fs *token.FileSet, s Stats) Stats {
	analyzer := &fileAnalyzer{
		file:    f,
		fileSet: fs,
		stats:   s,
	}
	return analyzer.analyze()
}

type fileAnalyzer struct {
	file    *ast.File
	fileSet *token.FileSet
	stats   Stats
}

func (a *fileAnalyzer) analyze() Stats {
	for _, decl := range a.file.Decls {
		a.analyzeDecl(decl)
	}
	return a.stats
}

func (a *fileAnalyzer) analyzeDecl(d ast.Decl) {
	switch decl := d.(type) {
	case *ast.FuncDecl:
		a.addStatIfNotIgnored(decl, funcName(decl), decl.Doc)
	case *ast.GenDecl:
		for _, spec := range decl.Specs {
			valueSpec, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			for _, value := range valueSpec.Values {
				funcLit, ok := value.(*ast.FuncLit)
				if !ok {
					continue
				}
				a.addStatIfNotIgnored(funcLit, valueSpec.Names[0].Name, decl.Doc)
			}
		}
	}
}

func (a *fileAnalyzer) addStatIfNotIgnored(node ast.Node, funcName string, doc *ast.CommentGroup) {
	if parseDirectives(doc).HasIgnore() {
		return
	}
	a.stats = append(a.stats, Stat{
		PkgName:    a.file.Name.Name,
		FuncName:   funcName,
		Complexity: Complexity(node),
		Pos:        a.fileSet.Position(node.Pos()),
	})
}

// funcName returns the name representation of a function or method:
// "(Type).Name" for methods or simply "Name" for functions.
func funcName(fn *ast.FuncDecl) string {
	if fn.Recv != nil {
		if fn.Recv.NumFields() > 0 {
			typ := fn.Recv.List[0].Type
			return fmt.Sprintf("(%s).%s", recvString(typ), fn.Name)
		}
	}
	return fn.Name.Name
}
