package golinters

import (
	"fmt"
	"sort"
	"sync"

	"github.com/uudashr/gocognit"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const gocognitName = "gocognit"

func NewGocognit() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: goanalysis.TheOnlyAnalyzerName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		gocognitName,
		"Computes and checks the cognitive complexity of functions",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			var stats []gocognit.Stat
			for _, f := range pass.Files {
				stats = gocognit.ComplexityStats(f, pass.Fset, stats)
			}
			if len(stats) == 0 {
				return nil, nil
			}

			sort.SliceStable(stats, func(i, j int) bool {
				return stats[i].Complexity > stats[j].Complexity
			})

			res := make([]goanalysis.Issue, 0, len(stats))
			for _, s := range stats {
				if s.Complexity <= lintCtx.Settings().Gocognit.MinComplexity {
					break // Break as the stats is already sorted from greatest to least
				}

				res = append(res, goanalysis.NewIssue(&result.Issue{
					Pos: s.Pos,
					Text: fmt.Sprintf("cognitive complexity %d of func %s is high (> %d)",
						s.Complexity, formatCode(s.FuncName, lintCtx.Cfg), lintCtx.Settings().Gocognit.MinComplexity),
					FromLinter: gocognitName,
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
