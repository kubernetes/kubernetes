// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modindex

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"iter"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"

	"golang.org/x/sync/errgroup"
)

// The name of a symbol contains information about the symbol:
// <name> T for types, TD if the type is deprecated
// <name> C for consts, CD if the const is deprecated
// <name> V for vars, VD if the var is deprecated
// and for funcs: <name> F <num of return values> (<arg-name> <arg-type>)*
// any spaces in <arg-type> are replaced by $s so that the fields
// of the name are space separated. F is replaced by FD if the func
// is deprecated.
type symbol struct {
	pkg  string // name of the symbols's package
	name string // declared name
	kind string // T, C, V, or F, followed by D if deprecated
	sig  string // signature information, for F
}

// extractSymbols returns a (new, unordered) array of Entries, one for
// each provided package directory, describing its exported symbols.
func extractSymbols(cwd string, dirs iter.Seq[directory]) []Entry {
	var (
		mu      sync.Mutex
		entries []Entry
	)

	var g errgroup.Group
	g.SetLimit(max(2, runtime.GOMAXPROCS(0)/2))
	for dir := range dirs {
		g.Go(func() error {
			thedir := filepath.Join(cwd, string(dir.path))
			mode := parser.SkipObjectResolution | parser.ParseComments

			// Parse all Go files in dir and extract symbols.
			dirents, err := os.ReadDir(thedir)
			if err != nil {
				return nil // log this someday?
			}
			var syms []symbol
			for _, dirent := range dirents {
				if !strings.HasSuffix(dirent.Name(), ".go") ||
					strings.HasSuffix(dirent.Name(), "_test.go") {
					continue
				}
				fname := filepath.Join(thedir, dirent.Name())
				tr, err := parser.ParseFile(token.NewFileSet(), fname, nil, mode)
				if err != nil {
					continue // ignore errors, someday log them?
				}
				syms = append(syms, getFileExports(tr)...)
			}

			// Create an entry for the package.
			pkg, names := processSyms(syms)
			if pkg != "" {
				mu.Lock()
				defer mu.Unlock()
				entries = append(entries, Entry{
					PkgName:    pkg,
					Dir:        dir.path,
					ImportPath: dir.importPath,
					Version:    dir.version,
					Names:      names,
				})
			}

			return nil
		})
	}
	g.Wait() // ignore error

	return entries
}

func getFileExports(f *ast.File) []symbol {
	pkg := f.Name.Name
	if pkg == "main" || pkg == "" {
		return nil
	}
	var ans []symbol
	// should we look for //go:build ignore?
	for _, decl := range f.Decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			if decl.Recv != nil {
				// ignore methods, as we are completing package selections
				continue
			}
			name := decl.Name.Name
			dtype := decl.Type
			// not looking at dtype.TypeParams. That is, treating
			// generic functions just like non-generic ones.
			sig := dtype.Params
			kind := "F"
			if isDeprecated(decl.Doc) {
				kind += "D"
			}
			result := []string{fmt.Sprintf("%d", dtype.Results.NumFields())}
			for _, x := range sig.List {
				// This code creates a string representing the type.
				// TODO(pjw): it may be fragile:
				// 1. x.Type could be nil, perhaps in ill-formed code
				// 2. ExprString might someday change incompatibly to
				//    include struct tags, which can be arbitrary strings
				if x.Type == nil {
					// Can this happen without a parse error? (Files with parse
					// errors are ignored in getSymbols)
					continue // maybe report this someday
				}
				tp := types.ExprString(x.Type)
				if len(tp) == 0 {
					// Can this happen?
					continue // maybe report this someday
				}
				// This is only safe if ExprString never returns anything with a $
				// The only place a $ can occur seems to be in a struct tag, which
				// can be an arbitrary string literal, and ExprString does not presently
				// print struct tags. So for this to happen the type of a formal parameter
				// has to be a explicit struct, e.g. foo(x struct{a int "$"}) and ExprString
				// would have to show the struct tag. Even testing for this case seems
				// a waste of effort, but let's remember the possibility
				if strings.Contains(tp, "$") {
					continue
				}
				tp = strings.Replace(tp, " ", "$", -1)
				if len(x.Names) == 0 {
					result = append(result, "_")
					result = append(result, tp)
				} else {
					for _, y := range x.Names {
						result = append(result, y.Name)
						result = append(result, tp)
					}
				}
			}
			sigs := strings.Join(result, " ")
			if s := newsym(pkg, name, kind, sigs); s != nil {
				ans = append(ans, *s)
			}
		case *ast.GenDecl:
			depr := isDeprecated(decl.Doc)
			switch decl.Tok {
			case token.CONST, token.VAR:
				tp := "V"
				if decl.Tok == token.CONST {
					tp = "C"
				}
				if depr {
					tp += "D"
				}
				for _, sp := range decl.Specs {
					for _, x := range sp.(*ast.ValueSpec).Names {
						if s := newsym(pkg, x.Name, tp, ""); s != nil {
							ans = append(ans, *s)
						}
					}
				}
			case token.TYPE:
				tp := "T"
				if depr {
					tp += "D"
				}
				for _, sp := range decl.Specs {
					if s := newsym(pkg, sp.(*ast.TypeSpec).Name.Name, tp, ""); s != nil {
						ans = append(ans, *s)
					}
				}
			}
		}
	}
	return ans
}

func newsym(pkg, name, kind, sig string) *symbol {
	if len(name) == 0 || !ast.IsExported(name) {
		return nil
	}
	sym := symbol{pkg: pkg, name: name, kind: kind, sig: sig}
	return &sym
}

func isDeprecated(doc *ast.CommentGroup) bool {
	if doc == nil {
		return false
	}
	// go.dev/wiki/Deprecated Paragraph starting 'Deprecated:'
	// This code fails for /* Deprecated: */, but it's the code from
	// gopls/internal/analysis/deprecated
	lines := strings.Split(doc.Text(), "\n\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "Deprecated:") {
			return true
		}
	}
	return false
}

// return the package name and the value for the symbols.
// if there are multiple packages, choose one arbitrarily
// the returned slice is sorted lexicographically
func processSyms(syms []symbol) (string, []string) {
	if len(syms) == 0 {
		return "", nil
	}
	slices.SortFunc(syms, func(l, r symbol) int {
		return strings.Compare(l.name, r.name)
	})
	pkg := syms[0].pkg
	var names []string
	for _, s := range syms {
		if s.pkg != pkg {
			// Symbols came from two files in same dir
			// with different package declarations.
			continue
		}
		var nx string
		if s.sig != "" {
			nx = fmt.Sprintf("%s %s %s", s.name, s.kind, s.sig)
		} else {
			nx = fmt.Sprintf("%s %s", s.name, s.kind)
		}
		names = append(names, nx)
	}
	return pkg, names
}
