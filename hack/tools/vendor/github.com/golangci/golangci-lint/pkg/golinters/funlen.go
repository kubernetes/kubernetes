package golinters

import (
	"go/token"
	"strings"
	"sync"

	"github.com/ultraware/funlen"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const funlenLinterName = "funlen"

func NewFunlen() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: funlenLinterName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		funlenLinterName,
		"Tool for detection of long functions",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			var issues []funlen.Message
			for _, file := range pass.Files {
				fileIssues := funlen.Run(file, pass.Fset, lintCtx.Settings().Funlen.Lines, lintCtx.Settings().Funlen.Statements)
				issues = append(issues, fileIssues...)
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
					FromLinter: funlenLinterName,
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
