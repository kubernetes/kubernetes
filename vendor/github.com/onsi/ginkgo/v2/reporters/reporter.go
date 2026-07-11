package reporters

import (
	"github.com/onsi/ginkgo/v2/types"
)

type Reporter interface {
	SuiteWillBegin(report types.Report)
	WillRun(report types.SpecReport)
	DidRun(report types.SpecReport)
	SuiteDidEnd(report types.Report)

	//Timeline emission
	EmitFailure(state types.SpecState, failure types.Failure)
	EmitProgressReport(progressReport types.ProgressReport)
	EmitReportEntry(entry types.ReportEntry)
	EmitSpecEvent(event types.SpecEvent)
}

type NoopReporter struct{}

func (n NoopReporter) SuiteWillBegin(report types.Report)                       {}
func (n NoopReporter) WillRun(report types.SpecReport)                          {}
func (n NoopReporter) DidRun(report types.SpecReport)                           {}
func (n NoopReporter) SuiteDidEnd(report types.Report)                          {}
func (n NoopReporter) EmitFailure(state types.SpecState, failure types.Failure) {}
func (n NoopReporter) EmitProgressReport(progressReport types.ProgressReport)   {}
func (n NoopReporter) EmitReportEntry(entry types.ReportEntry)                  {}
func (n NoopReporter) EmitSpecEvent(event types.SpecEvent)                      {}
