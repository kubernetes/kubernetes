package formatter

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/chavacava/garif"
	"github.com/mgechev/revive/lint"
)

// Sarif is an implementation of the Formatter interface
// which formats revive failures into SARIF format.
type Sarif struct {
	Metadata lint.FormatterMetadata
}

// Name returns the name of the formatter
func (f *Sarif) Name() string {
	return "sarif"
}

const reviveSite = "https://revive.run"

// Format formats the failures gotten from the lint.
func (f *Sarif) Format(failures <-chan lint.Failure, cfg lint.Config) (string, error) {
	sarifLog := newReviveRunLog(cfg)

	for failure := range failures {
		sarifLog.AddResult(failure)
	}

	buf := new(bytes.Buffer)
	sarifLog.PrettyWrite(buf)

	return buf.String(), nil
}

type reviveRunLog struct {
	*garif.LogFile
	run   *garif.Run
	rules map[string]lint.RuleConfig
}

func newReviveRunLog(cfg lint.Config) *reviveRunLog {
	run := garif.NewRun(garif.NewTool(garif.NewDriver("revive").WithInformationUri(reviveSite)))
	log := garif.NewLogFile([]*garif.Run{run}, garif.Version210)

	reviveLog := &reviveRunLog{
		log,
		run,
		cfg.Rules,
	}

	reviveLog.addRules(cfg.Rules)

	return reviveLog
}

func (l *reviveRunLog) addRules(cfg map[string]lint.RuleConfig) {
	for name, ruleCfg := range cfg {
		rule := garif.NewRule(name).WithHelpUri(reviveSite + "/r#" + name)
		setRuleProperties(rule, ruleCfg)
		driver := l.run.Tool.Driver

		if driver.Rules == nil {
			driver.Rules = []*garif.ReportingDescriptor{rule}
			return
		}

		driver.Rules = append(driver.Rules, rule)
	}
}

func (l *reviveRunLog) AddResult(failure lint.Failure) {
	positiveOrZero := func(x int) int {
		if x > 0 {
			return x
		}
		return 0
	}
	position := failure.Position
	filename := position.Start.Filename
	line := positiveOrZero(position.Start.Line - 1)     // https://docs.oasis-open.org/sarif/sarif/v2.1.0/csprd01/sarif-v2.1.0-csprd01.html#def_line
	column := positiveOrZero(position.Start.Column - 1) // https://docs.oasis-open.org/sarif/sarif/v2.1.0/csprd01/sarif-v2.1.0-csprd01.html#def_column

	result := garif.NewResult(garif.NewMessageFromText(failure.Failure))
	location := garif.NewLocation().WithURI(filename).WithLineColumn(line, column)
	result.Locations = append(result.Locations, location)
	result.RuleId = failure.RuleName
	result.Level = l.rules[failure.RuleName].Severity

	l.run.Results = append(l.run.Results, result)
}

func setRuleProperties(sarifRule *garif.ReportingDescriptor, lintRule lint.RuleConfig) {
	arguments := make([]string, len(lintRule.Arguments))
	for i, arg := range lintRule.Arguments {
		arguments[i] = fmt.Sprintf("%+v", arg)
	}

	if len(arguments) > 0 {
		sarifRule.WithProperties("arguments", strings.Join(arguments, ","))
	}

	sarifRule.WithProperties("severity", string(lintRule.Severity))
}
