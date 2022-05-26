// Package err113 is a Golang linter to check the errors handling expressions
package err113

import (
	"bytes"
	"go/ast"
	"go/printer"
	"go/token"

	"golang.org/x/tools/go/analysis"
)

// NewAnalyzer creates a new analysis.Analyzer instance tuned to run err113 checks.
func NewAnalyzer() *analysis.Analyzer {
	return &analysis.Analyzer{
		Name: "err113",
		Doc:  "checks the error handling rules according to the Go 1.13 new error type",
		Run:  run,
	}
}

func run(pass *analysis.Pass) (interface{}, error) {
	for _, file := range pass.Files {
		tlds := enumerateFileDecls(file)

		ast.Inspect(
			file,
			func(n ast.Node) bool {
				return inspectComparision(pass, n) &&
					inspectDefinition(pass, tlds, n)
			},
		)
	}

	return nil, nil
}

// render returns the pretty-print of the given node.
func render(fset *token.FileSet, x interface{}) string {
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, fset, x); err != nil {
		panic(err)
	}

	return buf.String()
}

func enumerateFileDecls(f *ast.File) map[*ast.CallExpr]struct{} {
	res := make(map[*ast.CallExpr]struct{})

	var ces []*ast.CallExpr // nolint: prealloc

	for _, d := range f.Decls {
		ces = append(ces, enumerateDeclVars(d)...)
	}

	for _, ce := range ces {
		res[ce] = struct{}{}
	}

	return res
}

func enumerateDeclVars(d ast.Decl) (res []*ast.CallExpr) {
	td, ok := d.(*ast.GenDecl)
	if !ok || td.Tok != token.VAR {
		return nil
	}

	for _, s := range td.Specs {
		res = append(res, enumerateSpecValues(s)...)
	}

	return res
}

func enumerateSpecValues(s ast.Spec) (res []*ast.CallExpr) {
	vs, ok := s.(*ast.ValueSpec)
	if !ok {
		return nil
	}

	for _, v := range vs.Values {
		if ce, ok := v.(*ast.CallExpr); ok {
			res = append(res, ce)
		}
	}

	return res
}
