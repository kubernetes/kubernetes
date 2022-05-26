package golinters

import (
	"go/token"
	"sync"

	"github.com/pkg/errors"
	"github.com/ultraware/whitespace"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

func NewWhitespace() *goanalysis.Linter {
	const linterName = "whitespace"
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: linterName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		linterName,
		"Tool for detection of leading and trailing whitespace",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		cfg := lintCtx.Cfg.LintersSettings.Whitespace
		settings := whitespace.Settings{MultiIf: cfg.MultiIf, MultiFunc: cfg.MultiFunc}

		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			var issues []whitespace.Message
			for _, file := range pass.Files {
				issues = append(issues, whitespace.Run(file, pass.Fset, settings)...)
			}

			if len(issues) == 0 {
				return nil, nil
			}

			res := make([]goanalysis.Issue, len(issues))
			for k, i := range issues {
				issue := result.Issue{
					Pos: token.Position{
						Filename: i.Pos.Filename,
						Line:     i.Pos.Line,
					},
					LineRange:   &result.Range{From: i.Pos.Line, To: i.Pos.Line},
					Text:        i.Message,
					FromLinter:  linterName,
					Replacement: &result.Replacement{},
				}

				bracketLine, err := lintCtx.LineCache.GetLine(issue.Pos.Filename, issue.Pos.Line)
				if err != nil {
					return nil, errors.Wrapf(err, "failed to get line %s:%d", issue.Pos.Filename, issue.Pos.Line)
				}

				switch i.Type {
				case whitespace.MessageTypeLeading:
					issue.LineRange.To++ // cover two lines by the issue: opening bracket "{" (issue.Pos.Line) and following empty line
				case whitespace.MessageTypeTrailing:
					issue.LineRange.From-- // cover two lines by the issue: closing bracket "}" (issue.Pos.Line) and preceding empty line
					issue.Pos.Line--       // set in sync with LineRange.From to not break fixer and other code features
				case whitespace.MessageTypeAddAfter:
					bracketLine += "\n"
				}
				issue.Replacement.NewLines = []string{bracketLine}

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
