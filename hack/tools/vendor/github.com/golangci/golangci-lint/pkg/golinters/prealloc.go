package golinters

import (
	"fmt"
	"sync"

	"github.com/alexkohler/prealloc/pkg"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const preallocName = "prealloc"

func NewPrealloc() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: preallocName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		preallocName,
		"Finds slice declarations that could potentially be preallocated",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		s := &lintCtx.Settings().Prealloc

		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			var res []goanalysis.Issue
			hints := pkg.Check(pass.Files, s.Simple, s.RangeLoops, s.ForLoops)
			for _, hint := range hints {
				res = append(res, goanalysis.NewIssue(&result.Issue{
					Pos:        pass.Fset.Position(hint.Pos),
					Text:       fmt.Sprintf("Consider preallocating %s", formatCode(hint.DeclaredSliceName, lintCtx.Cfg)),
					FromLinter: preallocName,
				}, pass))
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
