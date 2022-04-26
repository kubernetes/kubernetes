package processors

import (
	"fmt"
	"regexp"

	"github.com/golangci/golangci-lint/pkg/result"
)

type SkipFiles struct {
	patterns []*regexp.Regexp
}

var _ Processor = (*SkipFiles)(nil)

func NewSkipFiles(patterns []string) (*SkipFiles, error) {
	var patternsRe []*regexp.Regexp
	for _, p := range patterns {
		p = normalizePathInRegex(p)
		patternRe, err := regexp.Compile(p)
		if err != nil {
			return nil, fmt.Errorf("can't compile regexp %q: %s", p, err)
		}
		patternsRe = append(patternsRe, patternRe)
	}

	return &SkipFiles{
		patterns: patternsRe,
	}, nil
}

func (p SkipFiles) Name() string {
	return "skip_files"
}

func (p SkipFiles) Process(issues []result.Issue) ([]result.Issue, error) {
	if len(p.patterns) == 0 {
		return issues, nil
	}

	return filterIssues(issues, func(i *result.Issue) bool {
		for _, p := range p.patterns {
			if p.MatchString(i.FilePath()) {
				return false
			}
		}

		return true
	}), nil
}

func (p SkipFiles) Finish() {}
