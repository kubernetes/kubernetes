/*
Copyright 2019 The Kubernetes Authors.

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

// verify that all the imports have our preferred alias(es).
package main

import (
	"flag"
	"fmt"
	"github.com/google/go-cmp/cmp"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"golang.org/x/term"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

var (
	regex         = flag.String("include-path", "cmd/kubeadm", "only files with paths matching this regex is touched")
	isTerminal    = term.IsTerminal(int(os.Stdout.Fd()))
	logPrefix     = ""
	aliases       map[string]string
)

type analyzer struct {
	fset      *token.FileSet // positions are relative to fset
	ctx       build.Context
	failed    bool
	donePaths map[string]interface{}
}

func newAnalyzer() *analyzer {
	ctx := build.Default
	ctx.CgoEnabled = true

	a := &analyzer{
		fset:      token.NewFileSet(),
		ctx:       ctx,
		donePaths: make(map[string]interface{}),
	}

	return a
}

// collect extracts test metadata from a file.
func (a *analyzer) collect(dir string) {
	if _, ok := a.donePaths[dir]; ok {
		return
	}
	a.donePaths[dir] = nil

	// Create the AST by parsing src.
	fs, err := parser.ParseDir(a.fset, dir, nil, parser.AllErrors|parser.ParseComments)

	if err != nil {
		fmt.Fprintln(os.Stderr, "ERROR(syntax)", logPrefix, err)
		a.failed = true
		return
	}

	for _, p := range fs {
		// returns first error, but a.handleError deals with it
		files := a.filterFiles(p.Files)
		for _, file := range files {
			pathToFile := a.fset.File(file.Pos()).Name()

			var stdlibImports []string
			var localImports []string
			var k8sImports []string
			var externalImports []string
			var originalImports []string
			for _, imp := range file.Imports {
				importPath := strings.Replace(imp.Path.Value, "\"", "", -1)
				parts := strings.Split(importPath, "/")
				originalImports = append(originalImports, importPath)
				// standard library imports are first
				if !strings.Contains(parts[0], ".") {
					stdlibImports = append(stdlibImports, importPath)
				} else {
					if strings.Contains(parts[0], "k8s.io/kubernetes") {
						// local imports are second
						localImports = append(localImports, importPath)
					} else if strings.Contains(parts[0], "k8s.io") {
						// other *.k8s.io imports are third
						k8sImports = append(k8sImports, importPath)
					} else {
						// external repositories are fourth
						externalImports = append(externalImports, importPath)
					}
				}
			}
			if len(originalImports) == 0 {
				continue
			}
			sort.Strings(stdlibImports)
			sort.Strings(localImports)
			sort.Strings(k8sImports)
			sort.Strings(externalImports)
			allImports := []string{}
			allImports = append(allImports, stdlibImports...)
			allImports = append(allImports, localImports...)
			allImports = append(allImports, k8sImports...)
			allImports = append(allImports, externalImports...)
			if diff := cmp.Diff(originalImports, allImports); diff != "" {
				fmt.Printf("%s (-got +want):\n%s", pathToFile, diff)
			}
		}
	}
}

func (a *analyzer) filterFiles(fs map[string]*ast.File) []*ast.File {
	var files []*ast.File
	for _, f := range fs {
		files = append(files, f)
	}
	return files
}

type collector struct {
	dirs  []string
	regex *regexp.Regexp
}

// handlePath walks the filesystem recursively, collecting directories,
// ignoring some unneeded directories (hidden/vendored) that are handled
// specially later.
func (c *collector) handlePath(path string, info os.FileInfo, err error) error {
	if err != nil {
		return err
	}
	if info.IsDir() {
		// Ignore hidden directories (.git, .cache, etc)
		if len(path) > 1 && path[0] == '.' ||
			// Staging code is symlinked from vendor/k8s.io, and uses import
			// paths as if it were inside of vendor/. It fails typechecking
			// inside of staging/, but works when typechecked as part of vendor/.
			path == "staging" ||
			// OS-specific vendor code tends to be imported by OS-specific
			// packages. We recursively typecheck imported vendored packages for
			// each OS, but don't typecheck everything for every OS.
			path == "vendor" ||
			path == "_output" ||
			// This is a weird one. /testdata/ is *mostly* ignored by Go,
			// and this translates to kubernetes/vendor not working.
			// edit/record.go doesn't compile without gopkg.in/yaml.v2
			// in $GOSRC/$GOROOT (both typecheck and the shell script).
			path == "pkg/kubectl/cmd/testdata/edit" {
			return filepath.SkipDir
		}
		if c.regex.MatchString(path) {
			c.dirs = append(c.dirs, path)
		}
	}
	return nil
}

func main() {
	flag.Parse()
	args := flag.Args()

	if len(args) == 0 {
		args = append(args, ".")
	}

	regex, err := regexp.Compile(*regex)
	if err != nil {
		log.Fatalf("Error compiling regex: %v", err)
	}
	c := collector{regex: regex}
	for _, arg := range args {
		err := filepath.Walk(arg, c.handlePath)
		if err != nil {
			log.Fatalf("Error walking: %v", err)
		}
	}
	sort.Strings(c.dirs)

	if isTerminal {
		logPrefix = "\r" // clear status bar when printing
	}
	fmt.Println("checking-imports: ")

	a := newAnalyzer()
	for _, dir := range c.dirs {
		if isTerminal {
			fmt.Printf("\r\033[0m %-80s", dir)
		}
		a.collect(dir)
	}
	fmt.Println()
	if a.failed {
		os.Exit(1)
	}
}
