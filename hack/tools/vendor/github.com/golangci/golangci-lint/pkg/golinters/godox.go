package golinters

import (
	"go/token"
	"strings"
	"sync"

	"github.com/matoous/godox"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const godoxName = "godox"

func NewGodox() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: godoxName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		godoxName,
		"Tool for detection of FIXME, TODO and other comment keywords",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			var issues []godox.Message
			for _, file := range pass.Files {
				issues = append(issues, godox.Run(file, pass.Fset, lintCtx.Settings().Godox.Keywords...)...)
			}

			if len(issues) == 0 {
				return nil, nil
			}

			res := make([]goanalysis.Issue, len(issues))
			for k, i := range issues {
				res[k] = goanalysis.NewIssue(&result.Issue{
					Pos: token.Position{
						Filename: i.Pos.Filename,
						Line:     i.Pos.Line,
					},
					Text:       strings.TrimRight(i.Message, "\n"),
					FromLinter: godoxName,
				}, pass)
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
