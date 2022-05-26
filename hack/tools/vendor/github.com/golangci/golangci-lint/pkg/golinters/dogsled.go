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

const dogsledLinterName = "dogsled"

func NewDogsled() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: dogsledLinterName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		dogsledLinterName,
		"Checks assignments with too many blank identifiers (e.g. x, _, _, _, := f())",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			var pkgIssues []goanalysis.Issue
			for _, f := range pass.Files {
				v := returnsVisitor{
					maxBlanks: lintCtx.Settings().Dogsled.MaxBlankIdentifiers,
					f:         pass.Fset,
				}
				ast.Walk(&v, f)
				for i := range v.issues {
					pkgIssues = append(pkgIssues, goanalysis.NewIssue(&v.issues[i], pass))
				}
			}

			mu.Lock()
			resIssues = append(resIssues, pkgIssues...)
			mu.Unlock()

			return nil, nil
		}
	}).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return resIssues
	}).WithLoadMode(goanalysis.LoadModeSyntax)
}

type returnsVisitor struct {
	f         *token.FileSet
	maxBlanks int
	issues    []result.Issue
}

func (v *returnsVisitor) Visit(node ast.Node) ast.Visitor {
	funcDecl, ok := node.(*ast.FuncDecl)
	if !ok {
		return v
	}
	if funcDecl.Body == nil {
		return v
	}

	for _, expr := range funcDecl.Body.List {
		assgnStmt, ok := expr.(*ast.AssignStmt)
		if !ok {
			continue
		}

		numBlank := 0
		for _, left := range assgnStmt.Lhs {
			ident, ok := left.(*ast.Ident)
			if !ok {
				continue
			}
			if ident.Name == "_" {
				numBlank++
			}
		}

		if numBlank > v.maxBlanks {
			v.issues = append(v.issues, result.Issue{
				FromLinter: dogsledLinterName,
				Text:       fmt.Sprintf("declaration has %v blank identifiers", numBlank),
				Pos:        v.f.Position(assgnStmt.Pos()),
			})
		}
	}
	return v
}
