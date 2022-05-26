package golinters

import (
	"sync"

	goimportsAPI "github.com/golangci/gofmt/goimports"
	"github.com/pkg/errors"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/imports"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
)

const goimportsName = "goimports"

func NewGoimports() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: goimportsName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		goimportsName,
		"In addition to fixing imports, goimports also formats your code in the same style as gofmt.",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		imports.LocalPrefix = lintCtx.Settings().Goimports.LocalPrefixes
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			var fileNames []string
			for _, f := range pass.Files {
				pos := pass.Fset.PositionFor(f.Pos(), false)
				fileNames = append(fileNames, pos.Filename)
			}

			var issues []goanalysis.Issue

			for _, f := range fileNames {
				diff, err := goimportsAPI.Run(f)
				if err != nil { // TODO: skip
					return nil, err
				}
				if diff == nil {
					continue
				}

				is, err := extractIssuesFromPatch(string(diff), lintCtx.Log, lintCtx, goimportsName)
				if err != nil {
					return nil, errors.Wrapf(err, "can't extract issues from gofmt diff output %q", string(diff))
				}

				for i := range is {
					issues = append(issues, goanalysis.NewIssue(&is[i], pass))
				}
			}

			if len(issues) == 0 {
				return nil, nil
			}

			mu.Lock()
			resIssues = append(resIssues, issues...)
			mu.Unlock()

			return nil, nil
		}
	}).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return resIssues
	}).WithLoadMode(goanalysis.LoadModeSyntax)
}
