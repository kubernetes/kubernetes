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
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/format"
	"go/parser"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"golang.org/x/term"
)

var (
	importAliases = flag.String("import-aliases", "hack/.import-aliases", "json file with import aliases")
	confirm       = flag.Bool("confirm", false, "update file with the preferred aliases for imports")
	regex         = flag.String("include-path", "(test/e2e/|test/e2e_node)", "only files with paths matching this regex is touched")
	isTerminal    = term.IsTerminal(int(os.Stdout.Fd()))
	logPrefix     = ""
	aliases       = map[*regexp.Regexp]string{}
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
			replacements := make(map[string]string)
			pathToFile := a.fset.File(file.Pos()).Name()

			// Build the effective-alias map (alias → importPath) from the current
			// import declarations. It is updated as renames are applied so that
			// later imports in the same pass (or a subsequent pass) can see freed
			// aliases and don't falsely detect conflicts.
			effectiveAlias := make(map[string]string)
			for _, imp := range file.Imports {
				path := strings.Replace(imp.Path.Value, "\"", "", -1)
				segs := strings.Split(path, "/")
				name := segs[len(segs)-1]
				if imp.Name != nil {
					name = imp.Name.Name
				}
				if name != "_" && name != "." {
					effectiveAlias[name] = path
				}
			}

			// Iterate until no new renames are found. A single pass may leave some
			// imports unhandled because their target alias is temporarily held by
			// another import that will be freed in the same iteration, but after
			// the current position in the import list. Subsequent passes pick those
			// up once the conflicting alias has been released.
			for {
				iterCount := 0
				for _, imp := range file.Imports {
					importPath := strings.Replace(imp.Path.Value, "\"", "", -1)
					pathSegments := strings.Split(importPath, "/")
					importName := pathSegments[len(pathSegments)-1]
					if imp.Name != nil {
						importName = imp.Name.Name
					}
					for re, template := range aliases {
						match := re.FindStringSubmatchIndex(importPath)
						if match == nil {
							// No match.
							continue
						}
						if match[0] > 0 || match[1] < len(importPath) {
							// Not a full match.
							continue
						}
						alias := string(re.ExpandString(nil, template, importPath, match))
						if alias != importName {
							// Skip if the target alias is still in use by a different
							// import; it may be freed in a later pass.
							if existingPath, conflict := effectiveAlias[alias]; conflict && existingPath != importPath {
								break
							}
							if !*confirm {
								fmt.Fprintf(os.Stderr, "%sERROR wrong alias for import \"%s\" should be %s in file %s\n", logPrefix, importPath, alias, pathToFile)
								a.failed = true
							}
							replacements[importName] = alias
							delete(effectiveAlias, importName)
							effectiveAlias[alias] = importPath
							if imp.Name != nil {
								imp.Name.Name = alias
							} else {
								ident := ast.NewIdent(alias)
								ident.NamePos = imp.Path.Pos()
								imp.Name = ident
							}
							iterCount++
						}
						break
					}
				}
				if iterCount == 0 {
					break
				}
			}

			if len(replacements) > 0 {
				if *confirm {
					fmt.Printf("%sReplacing imports with aliases in file %s\n", logPrefix, pathToFile)
					renameImportUsages(file, replacements)
					ast.SortImports(a.fset, file)
					var buffer bytes.Buffer
					if err = format.Node(&buffer, a.fset, file); err != nil {
						panic(fmt.Sprintf("Error formatting ast node after rewriting import.\n%s\n", err.Error()))
					}

					fileInfo, err := os.Stat(pathToFile)
					if err != nil {
						panic(fmt.Sprintf("Error stat'ing file: %s\n%s\n", pathToFile, err.Error()))
					}

					err = os.WriteFile(pathToFile, buffer.Bytes(), fileInfo.Mode())
					if err != nil {
						panic(fmt.Sprintf("Error writing file: %s\n%s\n", pathToFile, err.Error()))
					}
				}
			}
		}
	}
}

// renameImportUsages applies all alias replacements in a single AST walk so
// that chained renames (A→B, B→C in the same file) don't cascade incorrectly.
// Import declarations are skipped because their alias identifiers are already
// updated by the caller before this function is invoked.
func renameImportUsages(f *ast.File, replacements map[string]string) {
	var pkg *ast.Ident
	ast.Inspect(f, func(node ast.Node) bool {
		if node == nil {
			return false
		}
		switch id := node.(type) {
		case *ast.File:
			pkg = id.Name
		case *ast.ImportSpec:
			return false
		case *ast.Ident:
			if pkg != nil && id == pkg {
				return false
			}
			if newName, ok := replacements[id.Name]; ok {
				id.Name = newName
			}
		}
		return true
	})
}

func (a *analyzer) filterFiles(fs map[string]*ast.File) []*ast.File {
	var files []*ast.File
	for _, f := range fs {
		if !isGeneratedFile(f) {
			files = append(files, f)
		}
	}
	return files
}

func isGeneratedFile(f *ast.File) bool {
	for _, cg := range f.Comments {
		for _, c := range cg.List {
			if strings.Contains(c.Text, "DO NOT EDIT") {
				return true
			}
		}
	}
	return false
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

	if len(*importAliases) > 0 {
		bytes, err := os.ReadFile(*importAliases)
		if err != nil {
			log.Fatalf("Error reading import aliases: %v", err)
		}
		var stringAliases map[string]string
		err = json.Unmarshal(bytes, &stringAliases)
		if err != nil {
			log.Fatalf("Error loading aliases: %v", err)
		}
		for pattern, name := range stringAliases {
			re, err := regexp.Compile(pattern)
			if err != nil {
				log.Fatalf("Error parsing import path pattern %q as regular expression: %v", pattern, err)
			}
			aliases[re] = name
		}
	}
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
