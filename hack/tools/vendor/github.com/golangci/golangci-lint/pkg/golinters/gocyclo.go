package golinters

import (
	"fmt"
	"sync"

	"github.com/fzipp/gocyclo"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const gocycloName = "gocyclo"

func NewGocyclo() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: gocycloName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		gocycloName,
		"Computes and checks the cyclomatic complexity of functions",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			var stats gocyclo.Stats
			for _, f := range pass.Files {
				stats = gocyclo.AnalyzeASTFile(f, pass.Fset, stats)
			}
			if len(stats) == 0 {
				return nil, nil
			}

			stats = stats.SortAndFilter(-1, lintCtx.Settings().Gocyclo.MinComplexity)

			res := make([]goanalysis.Issue, 0, len(stats))
			for _, s := range stats {
				res = append(res, goanalysis.NewIssue(&result.Issue{
					Pos: s.Pos,
					Text: fmt.Sprintf("cyclomatic complexity %d of func %s is high (> %d)",
						s.Complexity, formatCode(s.FuncName, lintCtx.Cfg), lintCtx.Settings().Gocyclo.MinComplexity),
					FromLinter: gocycloName,
				}, pass))
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
