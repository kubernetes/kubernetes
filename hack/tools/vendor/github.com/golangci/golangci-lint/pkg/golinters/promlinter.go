package golinters

import (
	"fmt"
	"sync"

	"github.com/yeya24/promlinter"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

func NewPromlinter() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	const linterName = "promlinter"
	analyzer := &analysis.Analyzer{
		Name: linterName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		linterName,
		"Check Prometheus metrics naming via promlint",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		strict := lintCtx.Cfg.LintersSettings.Promlinter.Strict
		disabledLinters := lintCtx.Cfg.LintersSettings.Promlinter.DisabledLinters

		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			issues := promlinter.RunLint(pass.Fset, pass.Files, promlinter.Setting{
				Strict:            strict,
				DisabledLintFuncs: disabledLinters,
			})

			if len(issues) == 0 {
				return nil, nil
			}

			res := make([]goanalysis.Issue, len(issues))
			for k, i := range issues {
				issue := result.Issue{
					Pos:        i.Pos,
					Text:       fmt.Sprintf("Metric: %s Error: %s", i.Metric, i.Text),
					FromLinter: linterName,
				}

				res[k] = goanalysis.NewIssue(&issue, pass)
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
