// Copyright (c) 2017, Daniel Martí <mvdan@mvdan.cc>
// See LICENSE for licensing information

package gogrep

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"strings"
	"text/template"
)

func transformSource(expr string) (string, []posOffset, error) {
	toks, err := tokenize([]byte(expr))
	if err != nil {
		return "", nil, fmt.Errorf("cannot tokenize expr: %v", err)
	}
	var offs []posOffset
	lbuf := lineColBuffer{line: 1, col: 1}
	lastLit := false
	for _, t := range toks {
		if lbuf.offs >= t.pos.Offset && lastLit && t.lit != "" {
			_, _ = lbuf.WriteString(" ")
		}
		for lbuf.offs < t.pos.Offset {
			_, _ = lbuf.WriteString(" ")
		}
		if t.lit == "" {
			_, _ = lbuf.WriteString(t.tok.String())
			lastLit = false
			continue
		}
		_, _ = lbuf.WriteString(t.lit)
		lastLit = strings.TrimSpace(t.lit) != ""
	}
	// trailing newlines can cause issues with commas
	return strings.TrimSpace(lbuf.String()), offs, nil
}

func parseExpr(fset *token.FileSet, expr string) (ast.Node, error) {
	exprStr, offs, err := transformSource(expr)
	if err != nil {
		return nil, err
	}
	node, _, err := parseDetectingNode(fset, exprStr)
	if err != nil {
		err = subPosOffsets(err, offs...)
		return nil, fmt.Errorf("cannot parse expr: %v", err)
	}
	return node, nil
}

type lineColBuffer struct {
	bytes.Buffer
	line, col, offs int
}

func (l *lineColBuffer) WriteString(s string) (n int, err error) {
	for _, r := range s {
		if r == '\n' {
			l.line++
			l.col = 1
		} else {
			l.col++
		}
		l.offs++
	}
	return l.Buffer.WriteString(s)
}

var tmplDecl = template.Must(template.New("").Parse(`` +
	`package p; {{ . }}`))

var tmplBlock = template.Must(template.New("").Parse(`` +
	`package p; func _() { if true {{ . }} else {} }`))

var tmplExprs = template.Must(template.New("").Parse(`` +
	`package p; var _ = []interface{}{ {{ . }}, }`))

var tmplStmts = template.Must(template.New("").Parse(`` +
	`package p; func _() { {{ . }} }`))

var tmplType = template.Must(template.New("").Parse(`` +
	`package p; var _ {{ . }}`))

var tmplValSpec = template.Must(template.New("").Parse(`` +
	`package p; var {{ . }}`))

func execTmpl(tmpl *template.Template, src string) string {
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, src); err != nil {
		panic(err)
	}
	return buf.String()
}

func noBadNodes(node ast.Node) bool {
	any := false
	ast.Inspect(node, func(n ast.Node) bool {
		if any {
			return false
		}
		switch n.(type) {
		case *ast.BadExpr, *ast.BadDecl:
			any = true
		}
		return true
	})
	return !any
}

func parseType(fset *token.FileSet, src string) (ast.Expr, *ast.File, error) {
	asType := execTmpl(tmplType, src)
	f, err := parser.ParseFile(fset, "", asType, 0)
	if err != nil {
		err = subPosOffsets(err, posOffset{1, 1, 17})
		return nil, nil, err
	}
	vs := f.Decls[0].(*ast.GenDecl).Specs[0].(*ast.ValueSpec)
	return vs.Type, f, nil
}

// parseDetectingNode tries its best to parse the ast.Node contained in src, as
// one of: *ast.File, ast.Decl, ast.Expr, ast.Stmt, *ast.ValueSpec.
// It also returns the *ast.File used for the parsing, so that the returned node
// can be easily type-checked.
func parseDetectingNode(fset *token.FileSet, src string) (ast.Node, *ast.File, error) {
	file := fset.AddFile("", fset.Base(), len(src))
	scan := scanner.Scanner{}
	scan.Init(file, []byte(src), nil, 0)
	if _, tok, _ := scan.Scan(); tok == token.EOF {
		return nil, nil, fmt.Errorf("empty source code")
	}
	var mainErr error

	// try as a block; otherwise blocks might be mistaken for composite
	// literals further below
	asBlock := execTmpl(tmplBlock, src)
	if f, err := parser.ParseFile(fset, "", asBlock, 0); err == nil && noBadNodes(f) {
		bl := f.Decls[0].(*ast.FuncDecl).Body
		if len(bl.List) == 1 {
			ifs := bl.List[0].(*ast.IfStmt)
			return ifs.Body, f, nil
		}
	}

	// then as value expressions
	asExprs := execTmpl(tmplExprs, src)
	if f, err := parser.ParseFile(fset, "", asExprs, 0); err == nil && noBadNodes(f) {
		vs := f.Decls[0].(*ast.GenDecl).Specs[0].(*ast.ValueSpec)
		cl := vs.Values[0].(*ast.CompositeLit)
		if len(cl.Elts) == 1 {
			return cl.Elts[0], f, nil
		}
		return ExprSlice(cl.Elts), f, nil
	}

	// then try as statements
	asStmts := execTmpl(tmplStmts, src)
	f, err := parser.ParseFile(fset, "", asStmts, 0)
	if err == nil && noBadNodes(f) {
		bl := f.Decls[0].(*ast.FuncDecl).Body
		if len(bl.List) == 1 {
			return bl.List[0], f, nil
		}
		return stmtSlice(bl.List), f, nil
	}
	// Statements is what covers most cases, so it will give
	// the best overall error message. Show positions
	// relative to where the user's code is put in the
	// template.
	mainErr = subPosOffsets(err, posOffset{1, 1, 22})

	// try as a single declaration, or many
	asDecl := execTmpl(tmplDecl, src)
	if f, err := parser.ParseFile(fset, "", asDecl, 0); err == nil && noBadNodes(f) {
		if len(f.Decls) == 1 {
			return f.Decls[0], f, nil
		}
		return declSlice(f.Decls), f, nil
	}

	// try as a whole file
	if f, err := parser.ParseFile(fset, "", src, 0); err == nil && noBadNodes(f) {
		return f, f, nil
	}

	// type expressions not yet picked up, for e.g. chans and interfaces
	if typ, f, err := parseType(fset, src); err == nil && noBadNodes(f) {
		return typ, f, nil
	}

	// value specs
	asValSpec := execTmpl(tmplValSpec, src)
	if f, err := parser.ParseFile(fset, "", asValSpec, 0); err == nil && noBadNodes(f) {
		vs := f.Decls[0].(*ast.GenDecl).Specs[0].(*ast.ValueSpec)
		return vs, f, nil
	}

	return nil, nil, mainErr
}

type posOffset struct {
	atLine, atCol int
	offset        int
}

func subPosOffsets(err error, offs ...posOffset) error {
	list, ok := err.(scanner.ErrorList)
	if !ok {
		return err
	}
	for i, err := range list {
		for _, off := range offs {
			if err.Pos.Line != off.atLine {
				continue
			}
			if err.Pos.Column < off.atCol {
				continue
			}
			err.Pos.Column -= off.offset
		}
		list[i] = err
	}
	return list
}

type fullToken struct {
	pos token.Position
	tok token.Token
	lit string
}

type caseStatus uint

const (
	caseNone caseStatus = iota
	caseNeedBlock
	caseHere
)

func tokenize(src []byte) ([]fullToken, error) {
	var s scanner.Scanner
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(src))

	var err error
	onError := func(pos token.Position, msg string) {
		switch msg { // allow certain extra chars
		case `illegal character U+0024 '$'`:
		case `illegal character U+007E '~'`:
		default:
			err = fmt.Errorf("%v: %s", pos, msg)
		}
	}

	// we will modify the input source under the scanner's nose to
	// enable some features such as regexes.
	s.Init(file, src, onError, scanner.ScanComments)

	next := func() fullToken {
		pos, tok, lit := s.Scan()
		return fullToken{fset.Position(pos), tok, lit}
	}

	caseStat := caseNone

	var toks []fullToken
	for t := next(); t.tok != token.EOF; t = next() {
		switch t.lit {
		case "$": // continues below
		case "switch", "select", "case":
			if t.lit == "case" {
				caseStat = caseNone
			} else {
				caseStat = caseNeedBlock
			}
			fallthrough
		default: // regular Go code
			if t.tok == token.LBRACE && caseStat == caseNeedBlock {
				caseStat = caseHere
			}
			toks = append(toks, t)
			continue
		}
		wt, err := tokenizeWildcard(t.pos, next)
		if err != nil {
			return nil, err
		}
		if caseStat == caseHere {
			toks = append(toks, fullToken{wt.pos, token.IDENT, "case"})
		}
		toks = append(toks, wt)
		if caseStat == caseHere {
			toks = append(toks,
				fullToken{wt.pos, token.COLON, ""},
				fullToken{wt.pos, token.IDENT, "gogrep_body"})
		}
	}
	return toks, err
}

type varInfo struct {
	Name string
	Seq  bool
}

func tokenizeWildcard(pos token.Position, next func() fullToken) (fullToken, error) {
	t := next()
	any := false
	if t.tok == token.MUL {
		t = next()
		any = true
	}
	wildName := encodeWildName(t.lit, any)
	wt := fullToken{pos, token.IDENT, wildName}
	if t.tok != token.IDENT {
		return wt, fmt.Errorf("%v: $ must be followed by ident, got %v",
			t.pos, t.tok)
	}
	return wt, nil
}

const wildSeparator = "ᐸᐳ"

func isWildName(s string) bool {
	return strings.HasPrefix(s, wildSeparator)
}

func encodeWildName(name string, any bool) string {
	suffix := "v"
	if any {
		suffix = "a"
	}
	return wildSeparator + name + wildSeparator + suffix
}

func decodeWildName(s string) varInfo {
	s = s[len(wildSeparator):]
	nameEnd := strings.Index(s, wildSeparator)
	name := s[:nameEnd+0]
	s = s[nameEnd:]
	s = s[len(wildSeparator):]
	kind := s
	return varInfo{Name: name, Seq: kind == "a"}
}

func decodeWildNode(n ast.Node) varInfo {
	switch n := n.(type) {
	case *ast.ExprStmt:
		return decodeWildNode(n.X)
	case *ast.Ident:
		if isWildName(n.Name) {
			return decodeWildName(n.Name)
		}
	}
	return varInfo{}
}
