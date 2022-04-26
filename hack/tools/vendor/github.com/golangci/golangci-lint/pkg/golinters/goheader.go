package golinters

import (
	"go/token"
	"sync"

	goheader "github.com/denis-tingaikin/go-header"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const goHeaderName = "goheader"

func NewGoHeader() *goanalysis.Linter {
	var mu sync.Mutex
	var issues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: goHeaderName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		goHeaderName,
		"Checks is file header matches to pattern",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		cfg := lintCtx.Cfg.LintersSettings.Goheader
		c := &goheader.Configuration{
			Values:       cfg.Values,
			Template:     cfg.Template,
			TemplatePath: cfg.TemplatePath,
		}
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			if c.TemplatePath == "" && c.Template == "" {
				// User did not pass template, so then do not run go-header linter
				return nil, nil
			}
			template, err := c.GetTemplate()
			if err != nil {
				return nil, err
			}
			values, err := c.GetValues()
			if err != nil {
				return nil, err
			}
			a := goheader.New(goheader.WithTemplate(template), goheader.WithValues(values))
			var res []goanalysis.Issue
			for _, file := range pass.Files {
				path := pass.Fset.Position(file.Pos()).Filename
				i := a.Analyze(&goheader.Target{
					File: file,
					Path: path,
				})
				if i == nil {
					continue
				}
				issue := result.Issue{
					Pos: token.Position{
						Line:     i.Location().Line + 1,
						Column:   i.Location().Position,
						Filename: path,
					},
					Text:       i.Message(),
					FromLinter: goHeaderName,
				}
				res = append(res, goanalysis.NewIssue(&issue, pass))
			}
			if len(res) == 0 {
				return nil, nil
			}

			mu.Lock()
			issues = append(issues, res...)
			mu.Unlock()

			return nil, nil
		}
	}).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return issues
	}).WithLoadMode(goanalysis.LoadModeSyntax)
}
