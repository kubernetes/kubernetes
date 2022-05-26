package goanalysis

import (
	"fmt"

	"github.com/pkg/errors"
	"golang.org/x/tools/go/packages"

	"github.com/golangci/golangci-lint/pkg/lint/linter"
	libpackages "github.com/golangci/golangci-lint/pkg/packages"
	"github.com/golangci/golangci-lint/pkg/result"
)

type IllTypedError struct {
	Pkg *packages.Package
}

func (e *IllTypedError) Error() string {
	return fmt.Sprintf("errors in package: %v", e.Pkg.Errors)
}

func buildIssuesFromIllTypedError(errs []error, lintCtx *linter.Context) ([]result.Issue, error) {
	var issues []result.Issue
	uniqReportedIssues := map[string]bool{}

	var other error

	for _, err := range errs {
		err := err

		var ill *IllTypedError
		if !errors.As(err, &ill) {
			if other == nil {
				other = err
			}
			continue
		}

		for _, err := range libpackages.ExtractErrors(ill.Pkg) {
			i, perr := parseError(err)
			if perr != nil { // failed to parse
				if uniqReportedIssues[err.Msg] {
					continue
				}
				uniqReportedIssues[err.Msg] = true
				lintCtx.Log.Errorf("typechecking error: %s", err.Msg)
			} else {
				i.Pkg = ill.Pkg // to save to cache later
				issues = append(issues, *i)
			}
		}
	}

	if len(issues) == 0 && other != nil {
		return nil, other
	}

	return issues, nil
}

func parseError(srcErr packages.Error) (*result.Issue, error) {
	pos, err := libpackages.ParseErrorPosition(srcErr.Pos)
	if err != nil {
		return nil, err
	}

	return &result.Issue{
		Pos:        *pos,
		Text:       srcErr.Msg,
		FromLinter: "typecheck",
	}, nil
}
