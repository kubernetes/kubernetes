// Copyright (c) 2013 The Go Authors. All rights reserved.
// Copyright (c) 2018 Dominik Honnef. All rights reserved.

package stylecheck

import (
	"fmt"
	"go/ast"
	"go/token"
	"strings"
	"unicode"

	"golang.org/x/tools/go/analysis"
	"honnef.co/go/tools/code"
	"honnef.co/go/tools/config"
	"honnef.co/go/tools/report"
)

// knownNameExceptions is a set of names that are known to be exempt from naming checks.
// This is usually because they are constrained by having to match names in the
// standard library.
var knownNameExceptions = map[string]bool{
	"LastInsertId": true, // must match database/sql
	"kWh":          true,
}

func CheckNames(pass *analysis.Pass) (interface{}, error) {
	// A large part of this function is copied from
	// github.com/golang/lint, Copyright (c) 2013 The Go Authors,
	// licensed under the BSD 3-clause license.

	allCaps := func(s string) bool {
		for _, r := range s {
			if !((r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_') {
				return false
			}
		}
		return true
	}

	check := func(id *ast.Ident, thing string, initialisms map[string]bool) {
		if id.Name == "_" {
			return
		}
		if knownNameExceptions[id.Name] {
			return
		}

		// Handle two common styles from other languages that don't belong in Go.
		if len(id.Name) >= 5 && allCaps(id.Name) && strings.Contains(id.Name, "_") {
			report.Report(pass, id, "should not use ALL_CAPS in Go names; use CamelCase instead", report.FilterGenerated())
			return
		}

		should := lintName(id.Name, initialisms)
		if id.Name == should {
			return
		}

		if len(id.Name) > 2 && strings.Contains(id.Name[1:len(id.Name)-1], "_") {
			report.Report(pass, id, fmt.Sprintf("should not use underscores in Go names; %s %s should be %s", thing, id.Name, should), report.FilterGenerated())
			return
		}
		report.Report(pass, id, fmt.Sprintf("%s %s should be %s", thing, id.Name, should), report.FilterGenerated())
	}
	checkList := func(fl *ast.FieldList, thing string, initialisms map[string]bool) {
		if fl == nil {
			return
		}
		for _, f := range fl.List {
			for _, id := range f.Names {
				check(id, thing, initialisms)
			}
		}
	}

	il := config.For(pass).Initialisms
	initialisms := make(map[string]bool, len(il))
	for _, word := range il {
		initialisms[word] = true
	}
	for _, f := range pass.Files {
		// Package names need slightly different handling than other names.
		if !strings.HasSuffix(f.Name.Name, "_test") && strings.Contains(f.Name.Name, "_") {
			report.Report(pass, f, "should not use underscores in package names", report.FilterGenerated())
		}
		if strings.IndexFunc(f.Name.Name, unicode.IsUpper) != -1 {
			report.Report(pass, f, fmt.Sprintf("should not use MixedCaps in package name; %s should be %s", f.Name.Name, strings.ToLower(f.Name.Name)), report.FilterGenerated())
		}
	}

	fn := func(node ast.Node) {
		switch v := node.(type) {
		case *ast.AssignStmt:
			if v.Tok != token.DEFINE {
				return
			}
			for _, exp := range v.Lhs {
				if id, ok := exp.(*ast.Ident); ok {
					check(id, "var", initialisms)
				}
			}
		case *ast.FuncDecl:
			// Functions with no body are defined elsewhere (in
			// assembly, or via go:linkname). These are likely to
			// be something very low level (such as the runtime),
			// where our rules don't apply.
			if v.Body == nil {
				return
			}

			if code.IsInTest(pass, v) && (strings.HasPrefix(v.Name.Name, "Example") || strings.HasPrefix(v.Name.Name, "Test") || strings.HasPrefix(v.Name.Name, "Benchmark")) {
				return
			}

			thing := "func"
			if v.Recv != nil {
				thing = "method"
			}

			if !isTechnicallyExported(v) {
				check(v.Name, thing, initialisms)
			}

			checkList(v.Type.Params, thing+" parameter", initialisms)
			checkList(v.Type.Results, thing+" result", initialisms)
		case *ast.GenDecl:
			if v.Tok == token.IMPORT {
				return
			}
			var thing string
			switch v.Tok {
			case token.CONST:
				thing = "const"
			case token.TYPE:
				thing = "type"
			case token.VAR:
				thing = "var"
			}
			for _, spec := range v.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					check(s.Name, thing, initialisms)
				case *ast.ValueSpec:
					for _, id := range s.Names {
						check(id, thing, initialisms)
					}
				}
			}
		case *ast.InterfaceType:
			// Do not check interface method names.
			// They are often constrained by the method names of concrete types.
			for _, x := range v.Methods.List {
				ft, ok := x.Type.(*ast.FuncType)
				if !ok { // might be an embedded interface name
					continue
				}
				checkList(ft.Params, "interface method parameter", initialisms)
				checkList(ft.Results, "interface method result", initialisms)
			}
		case *ast.RangeStmt:
			if v.Tok == token.ASSIGN {
				return
			}
			if id, ok := v.Key.(*ast.Ident); ok {
				check(id, "range var", initialisms)
			}
			if id, ok := v.Value.(*ast.Ident); ok {
				check(id, "range var", initialisms)
			}
		case *ast.StructType:
			for _, f := range v.Fields.List {
				for _, id := range f.Names {
					check(id, "struct field", initialisms)
				}
			}
		}
	}

	needle := []ast.Node{
		(*ast.AssignStmt)(nil),
		(*ast.FuncDecl)(nil),
		(*ast.GenDecl)(nil),
		(*ast.InterfaceType)(nil),
		(*ast.RangeStmt)(nil),
		(*ast.StructType)(nil),
	}

	code.Preorder(pass, fn, needle...)
	return nil, nil
}

// lintName returns a different name if it should be different.
func lintName(name string, initialisms map[string]bool) (should string) {
	// A large part of this function is copied from
	// github.com/golang/lint, Copyright (c) 2013 The Go Authors,
	// licensed under the BSD 3-clause license.

	// Fast path for simple cases: "_" and all lowercase.
	if name == "_" {
		return name
	}
	if strings.IndexFunc(name, func(r rune) bool { return !unicode.IsLower(r) }) == -1 {
		return name
	}

	// Split camelCase at any lower->upper transition, and split on underscores.
	// Check each word for common initialisms.
	runes := []rune(name)
	w, i := 0, 0 // index of start of word, scan
	for i+1 <= len(runes) {
		eow := false // whether we hit the end of a word
		if i+1 == len(runes) {
			eow = true
		} else if runes[i+1] == '_' && i+1 != len(runes)-1 {
			// underscore; shift the remainder forward over any run of underscores
			eow = true
			n := 1
			for i+n+1 < len(runes) && runes[i+n+1] == '_' {
				n++
			}

			// Leave at most one underscore if the underscore is between two digits
			if i+n+1 < len(runes) && unicode.IsDigit(runes[i]) && unicode.IsDigit(runes[i+n+1]) {
				n--
			}

			copy(runes[i+1:], runes[i+n+1:])
			runes = runes[:len(runes)-n]
		} else if unicode.IsLower(runes[i]) && !unicode.IsLower(runes[i+1]) {
			// lower->non-lower
			eow = true
		}
		i++
		if !eow {
			continue
		}

		// [w,i) is a word.
		word := string(runes[w:i])
		if u := strings.ToUpper(word); initialisms[u] {
			// Keep consistent case, which is lowercase only at the start.
			if w == 0 && unicode.IsLower(runes[w]) {
				u = strings.ToLower(u)
			}
			// All the common initialisms are ASCII,
			// so we can replace the bytes exactly.
			// TODO(dh): this won't be true once we allow custom initialisms
			copy(runes[w:], []rune(u))
		} else if w > 0 && strings.ToLower(word) == word {
			// already all lowercase, and not the first word, so uppercase the first character.
			runes[w] = unicode.ToUpper(runes[w])
		}
		w = i
	}
	return string(runes)
}

func isTechnicallyExported(f *ast.FuncDecl) bool {
	if f.Recv != nil || f.Doc == nil {
		return false
	}

	const export = "//export "
	const linkname = "//go:linkname "
	for _, c := range f.Doc.List {
		if strings.HasPrefix(c.Text, export) && len(c.Text) == len(export)+len(f.Name.Name) && c.Text[len(export):] == f.Name.Name {
			return true
		}

		if strings.HasPrefix(c.Text, linkname) {
			return true
		}
	}
	return false
}
