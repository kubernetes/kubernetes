package processors

import (
	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/logutils"
	"github.com/golangci/golangci-lint/pkg/result"
)

type MaxFromLinter struct {
	lc    linterToCountMap
	limit int
	log   logutils.Log
	cfg   *config.Config
}

var _ Processor = &MaxFromLinter{}

func NewMaxFromLinter(limit int, log logutils.Log, cfg *config.Config) *MaxFromLinter {
	return &MaxFromLinter{
		lc:    linterToCountMap{},
		limit: limit,
		log:   log,
		cfg:   cfg,
	}
}

func (p MaxFromLinter) Name() string {
	return "max_from_linter"
}

func (p *MaxFromLinter) Process(issues []result.Issue) ([]result.Issue, error) {
	if p.limit <= 0 { // no limit
		return issues, nil
	}

	return filterIssues(issues, func(i *result.Issue) bool {
		if i.Replacement != nil && p.cfg.Issues.NeedFix {
			// we need to fix all issues at once => we need to return all of them
			return true
		}

		p.lc[i.FromLinter]++ // always inc for stat
		return p.lc[i.FromLinter] <= p.limit
	}), nil
}

func (p MaxFromLinter) Finish() {
	walkStringToIntMapSortedByValue(p.lc, func(linter string, count int) {
		if count > p.limit {
			p.log.Infof("%d/%d issues from linter %s were hidden, use --max-issues-per-linter",
				count-p.limit, count, linter)
		}
	})
}
