package golinters

import (
	"fmt"
	"sync"

	"golang.org/x/tools/go/analysis"
	"honnef.co/go/tools/unused"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

type UnusedSettings struct {
	GoVersion string
}

func NewUnused(settings *config.StaticCheckSettings) *goanalysis.Linter {
	const name = "unused"

	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name:     name,
		Doc:      unused.Analyzer.Analyzer.Doc,
		Requires: unused.Analyzer.Analyzer.Requires,
		Run: func(pass *analysis.Pass) (interface{}, error) {
			res, err := unused.Analyzer.Analyzer.Run(pass)
			if err != nil {
				return nil, err
			}

			sr := unused.Serialize(pass, res.(unused.Result), pass.Fset)

			var issues []goanalysis.Issue
			for _, object := range sr.Unused {
				issue := goanalysis.NewIssue(&result.Issue{
					FromLinter: name,
					Text:       fmt.Sprintf("%s %s is unused", object.Kind, object.Name),
					Pos:        object.Position,
				}, pass)

				issues = append(issues, issue)
			}

			mu.Lock()
			resIssues = append(resIssues, issues...)
			mu.Unlock()

			return nil, nil
		},
	}

	setAnalyzerGoVersion(analyzer, getGoVersion(settings))

	lnt := goanalysis.NewLinter(
		name,
		"Checks Go code for unused constants, variables, functions and types",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithIssuesReporter(func(lintCtx *linter.Context) []goanalysis.Issue {
		return resIssues
	}).WithLoadMode(goanalysis.LoadModeTypesInfo)

	return lnt
}
