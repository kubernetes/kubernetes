package processors

import (
	"regexp"

	"github.com/golangci/golangci-lint/pkg/fsutils"
	"github.com/golangci/golangci-lint/pkg/logutils"
	"github.com/golangci/golangci-lint/pkg/result"
)

type excludeRule struct {
	baseRule
}

type ExcludeRule struct {
	BaseRule
}

type ExcludeRules struct {
	rules     []excludeRule
	lineCache *fsutils.LineCache
	log       logutils.Log
}

func NewExcludeRules(rules []ExcludeRule, lineCache *fsutils.LineCache, log logutils.Log) *ExcludeRules {
	r := &ExcludeRules{
		lineCache: lineCache,
		log:       log,
	}
	r.rules = createRules(rules, "(?i)")

	return r
}

func createRules(rules []ExcludeRule, prefix string) []excludeRule {
	parsedRules := make([]excludeRule, 0, len(rules))
	for _, rule := range rules {
		parsedRule := excludeRule{}
		parsedRule.linters = rule.Linters
		if rule.Text != "" {
			parsedRule.text = regexp.MustCompile(prefix + rule.Text)
		}
		if rule.Source != "" {
			parsedRule.source = regexp.MustCompile(prefix + rule.Source)
		}
		if rule.Path != "" {
			parsedRule.path = regexp.MustCompile(rule.Path)
		}
		parsedRules = append(parsedRules, parsedRule)
	}
	return parsedRules
}

func (p ExcludeRules) Process(issues []result.Issue) ([]result.Issue, error) {
	if len(p.rules) == 0 {
		return issues, nil
	}
	return filterIssues(issues, func(i *result.Issue) bool {
		for _, rule := range p.rules {
			rule := rule
			if rule.match(i, p.lineCache, p.log) {
				return false
			}
		}
		return true
	}), nil
}

func (ExcludeRules) Name() string { return "exclude-rules" }
func (ExcludeRules) Finish()      {}

var _ Processor = ExcludeRules{}

type ExcludeRulesCaseSensitive struct {
	*ExcludeRules
}

func NewExcludeRulesCaseSensitive(rules []ExcludeRule, lineCache *fsutils.LineCache, log logutils.Log) *ExcludeRulesCaseSensitive {
	r := &ExcludeRules{
		lineCache: lineCache,
		log:       log,
	}
	r.rules = createRules(rules, "")

	return &ExcludeRulesCaseSensitive{r}
}

func (ExcludeRulesCaseSensitive) Name() string { return "exclude-rules-case-sensitive" }

var _ Processor = ExcludeCaseSensitive{}
