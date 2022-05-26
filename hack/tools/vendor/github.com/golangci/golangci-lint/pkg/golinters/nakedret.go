package golinters

import (
	"fmt"
	"go/ast"
	"go/token"
	"sync"

	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

type nakedretVisitor struct {
	maxLength int
	f         *token.FileSet
	issues    []result.Issue
}

func (v *nakedretVisitor) processFuncDecl(funcDecl *ast.FuncDecl) {
	file := v.f.File(funcDecl.Pos())
	functionLineLength := file.Position(funcDecl.End()).Line - file.Position(funcDecl.Pos()).Line

	// Scan the body for usage of the named returns
	for _, stmt := range funcDecl.Body.List {
		s, ok := stmt.(*ast.ReturnStmt)
		if !ok {
			continue
		}

		if len(s.Results) != 0 {
			continue
		}

		file := v.f.File(s.Pos())
		if file == nil || functionLineLength <= v.maxLength {
			continue
		}
		if funcDecl.Name == nil {
			continue
		}

		v.issues = append(v.issues, result.Issue{
			FromLinter: nakedretName,
			Text: fmt.Sprintf("naked return in func `%s` with %d lines of code",
				funcDecl.Name.Name, functionLineLength),
			Pos: v.f.Position(s.Pos()),
		})
	}
}

func (v *nakedretVisitor) Visit(node ast.Node) ast.Visitor {
	funcDecl, ok := node.(*ast.FuncDecl)
	if !ok {
		return v
	}

	var namedReturns []*ast.Ident

	// We've found a function
	if funcDecl.Type != nil && funcDecl.Type.Results != nil {
		for _, field := range funcDecl.Type.Results.List {
			for _, ident := range field.Names {
				if ident != nil {
					namedReturns = append(namedReturns, ident)
				}
			}
		}
	}

	if len(namedReturns) == 0 || funcDecl.Body == nil {
		return v
	}

	v.processFuncDecl(funcDecl)
	return v
}

const nakedretName = "nakedret"

func NewNakedret() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: nakedretName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		nakedretName,
		"Finds naked returns in functions greater than a specified function length",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			var res []goanalysis.Issue
			for _, file := range pass.Files {
				v := nakedretVisitor{
					maxLength: lintCtx.Settings().Nakedret.MaxFuncLines,
					f:         pass.Fset,
				}
				ast.Walk(&v, file)
				for i := range v.issues {
					res = append(res, goanalysis.NewIssue(&v.issues[i], pass))
				}
			}

			if len(res) == 0 {
				return nil, nil
			}

			mu.Lock()
			resIssues = append(resIssues, res...)
			mu.Unlock()

			return nil, nil
		}
	}).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return resIssues
	}).WithLoadMode(goanalysis.LoadModeSyntax)
}
