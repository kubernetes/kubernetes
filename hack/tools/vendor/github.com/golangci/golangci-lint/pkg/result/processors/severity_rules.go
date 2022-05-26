package processors

import (
	"regexp"

	"github.com/golangci/golangci-lint/pkg/fsutils"
	"github.com/golangci/golangci-lint/pkg/logutils"
	"github.com/golangci/golangci-lint/pkg/result"
)

type severityRule struct {
	baseRule
	severity string
}

type SeverityRule struct {
	BaseRule
	Severity string
}

type SeverityRules struct {
	defaultSeverity string
	rules           []severityRule
	lineCache       *fsutils.LineCache
	log             logutils.Log
}

func NewSeverityRules(defaultSeverity string, rules []SeverityRule, lineCache *fsutils.LineCache, log logutils.Log) *SeverityRules {
	r := &SeverityRules{
		lineCache:       lineCache,
		log:             log,
		defaultSeverity: defaultSeverity,
	}
	r.rules = createSeverityRules(rules, "(?i)")

	return r
}

func createSeverityRules(rules []SeverityRule, prefix string) []severityRule {
	parsedRules := make([]severityRule, 0, len(rules))
	for _, rule := range rules {
		parsedRule := severityRule{}
		parsedRule.linters = rule.Linters
		parsedRule.severity = rule.Severity
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

func (p SeverityRules) Process(issues []result.Issue) ([]result.Issue, error) {
	if len(p.rules) == 0 && p.defaultSeverity == "" {
		return issues, nil
	}
	return transformIssues(issues, func(i *result.Issue) *result.Issue {
		for _, rule := range p.rules {
			rule := rule

			ruleSeverity := p.defaultSeverity
			if rule.severity != "" {
				ruleSeverity = rule.severity
			}

			if rule.match(i, p.lineCache, p.log) {
				i.Severity = ruleSeverity
				return i
			}
		}
		i.Severity = p.defaultSeverity
		return i
	}), nil
}

func (SeverityRules) Name() string { return "severity-rules" }
func (SeverityRules) Finish()      {}

var _ Processor = SeverityRules{}

type SeverityRulesCaseSensitive struct {
	*SeverityRules
}

func NewSeverityRulesCaseSensitive(defaultSeverity string, rules []SeverityRule,
	lineCache *fsutils.LineCache, log logutils.Log) *SeverityRulesCaseSensitive {
	r := &SeverityRules{
		lineCache:       lineCache,
		log:             log,
		defaultSeverity: defaultSeverity,
	}
	r.rules = createSeverityRules(rules, "")

	return &SeverityRulesCaseSensitive{r}
}

func (SeverityRulesCaseSensitive) Name() string { return "severity-rules-case-sensitive" }
