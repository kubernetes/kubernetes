package golinters

import (
	"fmt"
	"go/token"
	"strings"
	"sync"

	"github.com/golangci/misspell"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

func runMisspellOnFile(fileName string, r *misspell.Replacer, lintCtx *linter.Context) ([]result.Issue, error) {
	var res []result.Issue
	fileContent, err := lintCtx.FileCache.GetFileBytes(fileName)
	if err != nil {
		return nil, fmt.Errorf("can't get file %s contents: %s", fileName, err)
	}

	// use r.Replace, not r.ReplaceGo because r.ReplaceGo doesn't find
	// issues inside strings: it searches only inside comments. r.Replace
	// searches all words: it treats input as a plain text. A standalone misspell
	// tool uses r.Replace by default.
	_, diffs := r.Replace(string(fileContent))
	for _, diff := range diffs {
		text := fmt.Sprintf("`%s` is a misspelling of `%s`", diff.Original, diff.Corrected)
		pos := token.Position{
			Filename: fileName,
			Line:     diff.Line,
			Column:   diff.Column + 1,
		}
		replacement := &result.Replacement{
			Inline: &result.InlineFix{
				StartCol:  diff.Column,
				Length:    len(diff.Original),
				NewString: diff.Corrected,
			},
		}

		res = append(res, result.Issue{
			Pos:         pos,
			Text:        text,
			FromLinter:  misspellName,
			Replacement: replacement,
		})
	}

	return res, nil
}

const misspellName = "misspell"

func NewMisspell() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue
	var ruleErr error

	analyzer := &analysis.Analyzer{
		Name: misspellName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		misspellName,
		"Finds commonly misspelled English words in comments",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		r := misspell.Replacer{
			Replacements: misspell.DictMain,
		}

		// Figure out regional variations
		settings := lintCtx.Settings().Misspell
		locale := settings.Locale
		switch strings.ToUpper(locale) {
		case "":
			// nothing
		case "US":
			r.AddRuleList(misspell.DictAmerican)
		case "UK", "GB":
			r.AddRuleList(misspell.DictBritish)
		case "NZ", "AU", "CA":
			ruleErr = fmt.Errorf("unknown locale: %q", locale)
		}

		if ruleErr == nil {
			if len(settings.IgnoreWords) != 0 {
				r.RemoveRule(settings.IgnoreWords)
			}

			r.Compile()
		}

		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			if ruleErr != nil {
				return nil, ruleErr
			}

			var fileNames []string
			for _, f := range pass.Files {
				pos := pass.Fset.PositionFor(f.Pos(), false)
				fileNames = append(fileNames, pos.Filename)
			}

			var res []goanalysis.Issue
			for _, f := range fileNames {
				issues, err := runMisspellOnFile(f, &r, lintCtx)
				if err != nil {
					return nil, err
				}
				for i := range issues {
					res = append(res, goanalysis.NewIssue(&issues[i], pass))
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
	}).WithLoadMode(goanalysis.LoadModeSyntax)
}
