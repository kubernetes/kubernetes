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

const gochecknoinitsName = "gochecknoinits"

func NewGochecknoinits() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: gochecknoinitsName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
		Run: func(pass *analysis.Pass) (interface{}, error) {
			var res []goanalysis.Issue
			for _, file := range pass.Files {
				fileIssues := checkFileForInits(file, pass.Fset)
				for i := range fileIssues {
					res = append(res, goanalysis.NewIssue(&fileIssues[i], pass))
				}
			}
			if len(res) == 0 {
				return nil, nil
			}

			mu.Lock()
			resIssues = append(resIssues, res...)
			mu.Unlock()

			return nil, nil
		},
	}
	return goanalysis.NewLinter(
		gochecknoinitsName,
		"Checks that no init functions are present in Go code",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return resIssues
	}).WithLoadMode(goanalysis.LoadModeSyntax)
}

func checkFileForInits(f *ast.File, fset *token.FileSet) []result.Issue {
	var res []result.Issue
	for _, decl := range f.Decls {
		funcDecl, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}

		name := funcDecl.Name.Name
		if name == "init" && funcDecl.Recv.NumFields() == 0 {
			res = append(res, result.Issue{
				Pos:        fset.Position(funcDecl.Pos()),
				Text:       fmt.Sprintf("don't use %s function", formatCode(name, nil)),
				FromLinter: gochecknoinitsName,
			})
		}
	}

	return res
}
