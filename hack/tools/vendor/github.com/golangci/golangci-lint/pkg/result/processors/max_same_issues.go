package processors

import (
	"sort"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/logutils"
	"github.com/golangci/golangci-lint/pkg/result"
)

type textToCountMap map[string]int

type MaxSameIssues struct {
	tc    textToCountMap
	limit int
	log   logutils.Log
	cfg   *config.Config
}

var _ Processor = &MaxSameIssues{}

func NewMaxSameIssues(limit int, log logutils.Log, cfg *config.Config) *MaxSameIssues {
	return &MaxSameIssues{
		tc:    textToCountMap{},
		limit: limit,
		log:   log,
		cfg:   cfg,
	}
}

func (MaxSameIssues) Name() string {
	return "max_same_issues"
}

func (p *MaxSameIssues) Process(issues []result.Issue) ([]result.Issue, error) {
	if p.limit <= 0 { // no limit
		return issues, nil
	}

	return filterIssues(issues, func(i *result.Issue) bool {
		if i.Replacement != nil && p.cfg.Issues.NeedFix {
			// we need to fix all issues at once => we need to return all of them
			return true
		}

		p.tc[i.Text]++ // always inc for stat
		return p.tc[i.Text] <= p.limit
	}), nil
}

func (p MaxSameIssues) Finish() {
	walkStringToIntMapSortedByValue(p.tc, func(text string, count int) {
		if count > p.limit {
			p.log.Infof("%d/%d issues with text %q were hidden, use --max-same-issues",
				count-p.limit, count, text)
		}
	})
}

type kv struct {
	Key   string
	Value int
}

func walkStringToIntMapSortedByValue(m map[string]int, walk func(k string, v int)) {
	var ss []kv
	for k, v := range m {
		ss = append(ss, kv{
			Key:   k,
			Value: v,
		})
	}

	sort.Slice(ss, func(i, j int) bool {
		return ss[i].Value > ss[j].Value
	})

	for _, kv := range ss {
		walk(kv.Key, kv.Value)
	}
}
