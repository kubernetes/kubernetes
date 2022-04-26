package golinters

import (
	"fmt"
	"sync"

	malignedAPI "github.com/golangci/maligned"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

func NewMaligned() *goanalysis.Linter {
	const linterName = "maligned"
	var mu sync.Mutex
	var res []goanalysis.Issue
	analyzer := &analysis.Analyzer{
		Name: linterName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		linterName,
		"Tool to detect Go structs that would take less memory if their fields were sorted",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			prog := goanalysis.MakeFakeLoaderProgram(pass)

			malignedIssues := malignedAPI.Run(prog)
			if len(malignedIssues) == 0 {
				return nil, nil
			}

			issues := make([]goanalysis.Issue, 0, len(malignedIssues))
			for _, i := range malignedIssues {
				text := fmt.Sprintf("struct of size %d bytes could be of size %d bytes", i.OldSize, i.NewSize)
				if lintCtx.Settings().Maligned.SuggestNewOrder {
					text += fmt.Sprintf(":\n%s", formatCodeBlock(i.NewStructDef, lintCtx.Cfg))
				}
				issues = append(issues, goanalysis.NewIssue(&result.Issue{
					Pos:        i.Pos,
					Text:       text,
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
