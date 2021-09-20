package reporters

import (
	"github.com/onsi/ginkgo/v2/types"
)

type Reporter interface {
	SuiteWillBegin(report types.Report)
	WillRun(report types.SpecReport)
	DidRun(report types.SpecReport)
	SuiteDidEnd(report types.Report)
}

type NoopReporter struct{}

func (n NoopReporter) SuiteWillBegin(report types.Report) {}
func (n NoopReporter) WillRun(report types.SpecReport)    {}
func (n NoopReporter) DidRun(report types.SpecReport)     {}
func (n NoopReporter) SuiteDidEnd(report types.Report)    {}
