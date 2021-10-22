// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

// mkmerge.go parses generated source files and merges common
// consts, funcs, and types into a common source file, per GOOS.
//
// Usage:
//     $ go run mkmerge.go -out MERGED FILE [FILE ...]
//
// Example:
//     # Remove all common consts, funcs, and types from zerrors_linux_*.go
//     # and write the common code into zerrors_linux.go
//     $ go run mkmerge.go -out zerrors_linux.go zerrors_linux_*.go
//
// mkmerge.go performs the merge in the following steps:
// 1. Construct the set of common code that is idential in all
//    architecture-specific files.
// 2. Write this common code to the merged file.
// 3. Remove the common code from all architecture-specific files.
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

const validGOOS = "aix|darwin|dragonfly|freebsd|linux|netbsd|openbsd|solaris"

// getValidGOOS returns GOOS, true if filename ends with a valid "_GOOS.go"
func getValidGOOS(filename string) (string, bool) {
	matches := regexp.MustCompile(`_(` + validGOOS + `)\.go$`).FindStringSubmatch(filename)
	if len(matches) != 2 {
		return "", false
	}
	return matches[1], true
}

// codeElem represents an ast.Decl in a comparable way.
type codeElem struct {
	tok token.Token // e.g. token.CONST, token.TYPE, or token.FUNC
	src string      // the declaration formatted as source code
}

// newCodeElem returns a codeElem based on tok and node, or an error is returned.
func newCodeElem(tok token.Token, node ast.Node) (codeElem, error) {
	var b strings.Builder
	err := format.Node(&b, token.NewFileSet(), node)
	if err != nil {
		return codeElem{}, err
	}
	return codeElem{tok, b.String()}, nil
}

// codeSet is a set of codeElems
type codeSet struct {
	set map[codeElem]bool // true for all codeElems in the set
}

// newCodeSet returns a new codeSet
func newCodeSet() *codeSet { return &codeSet{make(map[codeElem]bool)} }

// add adds elem to c
func (c *codeSet) add(elem codeElem) { c.set[elem] = true }

// has returns true if elem is in c
func (c *codeSet) has(elem codeElem) bool { return c.set[elem] }

// isEmpty returns true if the set is empty
func (c *codeSet) isEmpty() bool { return len(c.set) == 0 }

// intersection returns a new set which is the intersection of c and a
func (c *codeSet) intersection(a *codeSet) *codeSet {
	res := newCodeSet()

	for elem := range c.set {
		if a.has(elem) {
			res.add(elem)
		}
	}
	return res
}

// keepCommon is a filterFn for filtering the merged file with common declarations.
func (c *codeSet) keepCommon(elem codeElem) bool {
	switch elem.tok {
	case token.VAR:
		// Remove all vars from the merged file
		return false
	case token.CONST, token.TYPE, token.FUNC, token.COMMENT:
		// Remove arch-specific consts, types, functions, and file-level comments from the merged file
		return c.has(elem)
	case token.IMPORT:
		// Keep imports, they are handled by filterImports
		return true
	}

	log.Fatalf("keepCommon: invalid elem %v", elem)
	return true
}

// keepArchSpecific is a filterFn for filtering the GOARC-specific files.
func (c *codeSet) keepArchSpecific(elem codeElem) bool {
	switch elem.tok {
	case token.CONST, token.TYPE, token.FUNC:
		// Remove common consts, types, or functions from the arch-specific file
		return !c.has(elem)
	}
	return true
}

// srcFile represents a source file
type srcFile struct {
	name string
	src  []byte
}

// filterFn is a helper for filter
type filterFn func(codeElem) bool

// filter parses and filters Go source code from src, removing top
// level declarations using keep as predicate.
// For src parameter, please see docs for parser.ParseFile.
func filter(src interface{}, keep filterFn) ([]byte, error) {
	// Parse the src into an ast
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	cmap := ast.NewCommentMap(fset, f, f.Comments)

	// Group const/type specs on adjacent lines
	var groups specGroups = make(map[string]int)
	var groupID int

	decls := f.Decls
	f.Decls = f.Decls[:0]
	for _, decl := range decls {
		switch decl := decl.(type) {
		case *ast.GenDecl:
			// Filter imports, consts, types, vars
			specs := decl.Specs
			decl.Specs = decl.Specs[:0]
			for i, spec := range specs {
				elem, err := newCodeElem(decl.Tok, spec)
				if err != nil {
					return nil, err
				}

				// Create new group if there are empty lines between this and the previous spec
				if i > 0 && fset.Position(specs[i-1].End()).Line < fset.Position(spec.Pos()).Line-1 {
					groupID++
				}

				// Check if we should keep this spec
				if keep(elem) {
					decl.Specs = append(decl.Specs, spec)
					groups.add(elem.src, groupID)
				}
			}
			// Check if we should keep this decl
			if len(decl.Specs) > 0 {
				f.Decls = append(f.Decls, decl)
			}
		case *ast.FuncDecl:
			// Filter funcs
			elem, err := newCodeElem(token.FUNC, decl)
			if err != nil {
				return nil, err
			}
			if keep(elem) {
				f.Decls = append(f.Decls, decl)
			}
		}
	}

	// Filter file level comments
	if cmap[f] != nil {
		commentGroups := cmap[f]
		cmap[f] = cmap[f][:0]
		for _, cGrp := range commentGroups {
			if keep(codeElem{token.COMMENT, cGrp.Text()}) {
				cmap[f] = append(cmap[f], cGrp)
			}
		}
	}
	f.Comments = cmap.Filter(f).Comments()

	// Generate code for the filtered ast
	var buf bytes.Buffer
	if err = format.Node(&buf, fset, f); err != nil {
		return nil, err
	}

	groupedSrc, err := groups.filterEmptyLines(&buf)
	if err != nil {
		return nil, err
	}

	return filterImports(groupedSrc)
}

// getCommonSet returns the set of consts, types, and funcs that are present in every file.
func getCommonSet(files []srcFile) (*codeSet, error) {
	if len(files) == 0 {
		return nil, fmt.Errorf("no files provided")
	}
	// Use the first architecture file as the baseline
	baseSet, err := getCodeSet(files[0].src)
	if err != nil {
		return nil, err
	}

	// Compare baseline set with other architecture files: discard any element,
	// that doesn't exist in other architecture files.
	for _, f := range files[1:] {
		set, err := getCodeSet(f.src)
		if err != nil {
			return nil, err
		}

		baseSet = baseSet.intersection(set)
	}
	return baseSet, nil
}

// getCodeSet returns the set of all top-level consts, types, and funcs from src.
// src must be string, []byte, or io.Reader (see go/parser.ParseFile docs)
func getCodeSet(src interface{}) (*codeSet, error) {
	set := newCodeSet()

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	for _, decl := range f.Decls {
		switch decl := decl.(type) {
		case *ast.GenDecl:
			// Add const, and type declarations
			if !(decl.Tok == token.CONST || decl.Tok == token.TYPE) {
				break
			}

			for _, spec := range decl.Specs {
				elem, err := newCodeElem(decl.Tok, spec)
				if err != nil {
					return nil, err
				}

				set.add(elem)
			}
		case *ast.FuncDecl:
			// Add func declarations
			elem, err := newCodeElem(token.FUNC, decl)
			if err != nil {
				return nil, err
			}

			set.add(elem)
		}
	}

	// Add file level comments
	cmap := ast.NewCommentMap(fset, f, f.Comments)
	for _, cGrp := range cmap[f] {
		set.add(codeElem{token.COMMENT, cGrp.Text()})
	}

	return set, nil
}

// importName returns the identifier (PackageName) for an imported package
func importName(iSpec *ast.ImportSpec) (string, error) {
	if iSpec.Name == nil {
		name, err := strconv.Unquote(iSpec.Path.Value)
		if err != nil {
			return "", err
		}
		return path.Base(name), nil
	}
	return iSpec.Name.Name, nil
}

// specGroups tracks grouped const/type specs with a map of line: groupID pairs
type specGroups map[string]int

// add spec source to group
func (s specGroups) add(src string, groupID int) error {
	srcBytes, err := format.Source(bytes.TrimSpace([]byte(src)))
	if err != nil {
		return err
	}
	s[string(srcBytes)] = groupID
	return nil
}

// filterEmptyLines removes empty lines within groups of const/type specs.
// Returns the filtered source.
func (s specGroups) filterEmptyLines(src io.Reader) ([]byte, error) {
	scanner := bufio.NewScanner(src)
	var out bytes.Buffer

	var emptyLines bytes.Buffer
	prevGroupID := -1 // Initialize to invalid group
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())

		if len(line) == 0 {
			fmt.Fprintf(&emptyLines, "%s\n", scanner.Bytes())
			continue
		}

		// Discard emptyLines if previous non-empty line belonged to the same
		// group as this line
		if src, err := format.Source(line); err == nil {
			groupID, ok := s[string(src)]
			if ok && groupID == prevGroupID {
				emptyLines.Reset()
			}
			prevGroupID = groupID
		}

		emptyLines.WriteTo(&out)
		fmt.Fprintf(&out, "%s\n", scanner.Bytes())
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

// filterImports removes unused imports from fileSrc, and returns a formatted src.
func filterImports(fileSrc []byte) ([]byte, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", fileSrc, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	cmap := ast.NewCommentMap(fset, file, file.Comments)

	// create set of references to imported identifiers
	keepImport := make(map[string]bool)
	for _, u := range file.Unresolved {
		keepImport[u.Name] = true
	}

	// filter import declarations
	decls := file.Decls
	file.Decls = file.Decls[:0]
	for _, decl := range decls {
		importDecl, ok := decl.(*ast.GenDecl)

		// Keep non-import declarations
		if !ok || importDecl.Tok != token.IMPORT {
			file.Decls = append(file.Decls, decl)
			continue
		}

		// Filter the import specs
		specs := importDecl.Specs
		importDecl.Specs = importDecl.Specs[:0]
		for _, spec := range specs {
			iSpec := spec.(*ast.ImportSpec)
			name, err := importName(iSpec)
			if err != nil {
				return nil, err
			}

			if keepImport[name] {
				importDecl.Specs = append(importDecl.Specs, iSpec)
			}
		}
		if len(importDecl.Specs) > 0 {
			file.Decls = append(file.Decls, importDecl)
		}
	}

	// filter file.Imports
	imports := file.Imports
	file.Imports = file.Imports[:0]
	for _, spec := range imports {
		name, err := importName(spec)
		if err != nil {
			return nil, err
		}

		if keepImport[name] {
			file.Imports = append(file.Imports, spec)
		}
	}
	file.Comments = cmap.Filter(file).Comments()

	var buf bytes.Buffer
	err = format.Node(&buf, fset, file)
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// merge extracts duplicate code from archFiles and merges it to mergeFile.
// 1. Construct commonSet: the set of code that is idential in all archFiles.
// 2. Write the code in commonSet to mergedFile.
// 3. Remove the commonSet code from all archFiles.
func merge(mergedFile string, archFiles ...string) error {
	// extract and validate the GOOS part of the merged filename
	goos, ok := getValidGOOS(mergedFile)
	if !ok {
		return fmt.Errorf("invalid GOOS in merged file name %s", mergedFile)
	}

	// Read architecture files
	var inSrc []srcFile
	for _, file := range archFiles {
		src, err := ioutil.ReadFile(file)
		if err != nil {
			return fmt.Errorf("cannot read archfile %s: %w", file, err)
		}

		inSrc = append(inSrc, srcFile{file, src})
	}

	// 1. Construct the set of top-level declarations common for all files
	commonSet, err := getCommonSet(inSrc)
	if err != nil {
		return err
	}
	if commonSet.isEmpty() {
		// No common code => do not modify any files
		return nil
	}

	// 2. Write the merged file
	mergedSrc, err := filter(inSrc[0].src, commonSet.keepCommon)
	if err != nil {
		return err
	}

	f, err := os.Create(mergedFile)
	if err != nil {
		return err
	}

	buf := bufio.NewWriter(f)
	fmt.Fprintln(buf, "// Code generated by mkmerge.go; DO NOT EDIT.")
	fmt.Fprintln(buf)
	fmt.Fprintf(buf, "//go:build %s\n", goos)
	fmt.Fprintf(buf, "// +build %s\n", goos)
	fmt.Fprintln(buf)
	buf.Write(mergedSrc)

	err = buf.Flush()
	if err != nil {
		return err
	}
	err = f.Close()
	if err != nil {
		return err
	}

	// 3. Remove duplicate declarations from the architecture files
	for _, inFile := range inSrc {
		src, err := filter(inFile.src, commonSet.keepArchSpecific)
		if err != nil {
			return err
		}
		err = ioutil.WriteFile(inFile.name, src, 0644)
		if err != nil {
			return err
		}
	}
	return nil
}

func main() {
	var mergedFile string
	flag.StringVar(&mergedFile, "out", "", "Write merged code to `FILE`")
	flag.Parse()

	// Expand wildcards
	var filenames []string
	for _, arg := range flag.Args() {
		matches, err := filepath.Glob(arg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Invalid command line argument %q: %v\n", arg, err)
			os.Exit(1)
		}
		filenames = append(filenames, matches...)
	}

	if len(filenames) < 2 {
		// No need to merge
		return
	}

	err := merge(mergedFile, filenames...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Merge failed with error: %v\n", err)
		os.Exit(1)
	}
}
