package golinters

import (
	"sync"

	"github.com/ashanbrown/makezero/makezero"
	"github.com/pkg/errors"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const makezeroName = "makezero"

func NewMakezero() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: makezeroName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		makezeroName,
		"Finds slice declarations with non-zero initial length",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		s := &lintCtx.Settings().Makezero

		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			var res []goanalysis.Issue
			linter := makezero.NewLinter(s.Always)
			for _, file := range pass.Files {
				hints, err := linter.Run(pass.Fset, pass.TypesInfo, file)
				if err != nil {
					return nil, errors.Wrapf(err, "makezero linter failed on file %q", file.Name.String())
				}
				for _, hint := range hints {
					res = append(res, goanalysis.NewIssue(&result.Issue{
						Pos:        hint.Position(),
						Text:       hint.Details(),
						FromLinter: makezeroName,
					}, pass))
				}
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
	}).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
