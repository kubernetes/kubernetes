package golinters

import (
	"sync"

	unconvertAPI "github.com/golangci/unconvert"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

func NewUnconvert() *goanalysis.Linter {
	const linterName = "unconvert"
	var mu sync.Mutex
	var res []goanalysis.Issue
	analyzer := &analysis.Analyzer{
		Name: linterName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		linterName,
		"Remove unnecessary type conversions",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			prog := goanalysis.MakeFakeLoaderProgram(pass)

			positions := unconvertAPI.Run(prog)
			if len(positions) == 0 {
				return nil, nil
			}

			issues := make([]goanalysis.Issue, 0, len(positions))
			for _, pos := range positions {
				issues = append(issues, goanalysis.NewIssue(&result.Issue{
					Pos:        pos,
					Text:       "unnecessary conversion",
					FromLinter: linterName,
				}, pass))
			}

			mu.Lock()
			res = append(res, issues...)
			mu.Unlock()
			return nil, nil
		}
	}).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return res
	}).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
