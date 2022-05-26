package golinters

import (
	"sort"
	"sync"

	"github.com/nakabonne/nestif"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const nestifName = "nestif"

func NewNestif() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: goanalysis.TheOnlyAnalyzerName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		nestifName,
		"Reports deeply nested if statements",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			checker := &nestif.Checker{
				MinComplexity: lintCtx.Settings().Nestif.MinComplexity,
			}
			var issues []nestif.Issue
			for _, f := range pass.Files {
				issues = append(issues, checker.Check(f, pass.Fset)...)
			}
			if len(issues) == 0 {
				return nil, nil
			}

			sort.SliceStable(issues, func(i, j int) bool {
				return issues[i].Complexity > issues[j].Complexity
			})

			res := make([]goanalysis.Issue, 0, len(issues))
			for _, i := range issues {
				res = append(res, goanalysis.NewIssue(&result.Issue{
					Pos:        i.Pos,
					Text:       i.Message,
					FromLinter: nestifName,
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
