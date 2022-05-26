package processors

import (
	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/result"
)

type lineToCount map[int]int
type fileToLineToCount map[string]lineToCount

type UniqByLine struct {
	flc fileToLineToCount
	cfg *config.Config
}

func NewUniqByLine(cfg *config.Config) *UniqByLine {
	return &UniqByLine{
		flc: fileToLineToCount{},
		cfg: cfg,
	}
}

var _ Processor = &UniqByLine{}

func (p UniqByLine) Name() string {
	return "uniq_by_line"
}

func (p *UniqByLine) Process(issues []result.Issue) ([]result.Issue, error) {
	if !p.cfg.Output.UniqByLine {
		return issues, nil
	}

	return filterIssues(issues, func(i *result.Issue) bool {
		if i.Replacement != nil && p.cfg.Issues.NeedFix {
			// if issue will be auto-fixed we shouldn't collapse issues:
			// e.g. one line can contain 2 misspellings, they will be in 2 issues and misspell should fix both of them.
			return true
		}

		lc := p.flc[i.FilePath()]
		if lc == nil {
			lc = lineToCount{}
			p.flc[i.FilePath()] = lc
		}

		const limit = 1
		count := lc[i.Line()]
		if count == limit {
			return false
		}

		lc[i.Line()]++
		return true
	}), nil
}

func (p UniqByLine) Finish() {}
