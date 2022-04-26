package formatter

import "github.com/mgechev/revive/lint"

func severity(config lint.Config, failure lint.Failure) lint.Severity {
	if config, ok := config.Rules[failure.RuleName]; ok && config.Severity == lint.SeverityError {
		return lint.SeverityError
	}
	if config, ok := config.Directives[failure.RuleName]; ok && config.Severity == lint.SeverityError {
		return lint.SeverityError
	}
	return lint.SeverityWarning
}
