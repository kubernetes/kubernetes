package reporters

import (
	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/types"
)

//FakeReporter is useful for testing purposes
type FakeReporter struct {
	Config config.GinkgoConfigType

	BeginSummary         *types.SuiteSummary
	BeforeSuiteSummary   *types.SetupSummary
	SpecWillRunSummaries []*types.SpecSummary
	SpecSummaries        []*types.SpecSummary
	AfterSuiteSummary    *types.SetupSummary
	EndSummary           *types.SuiteSummary

	SpecWillRunStub     func(specSummary *types.SpecSummary)
	SpecDidCompleteStub func(specSummary *types.SpecSummary)
}

func NewFakeReporter() *FakeReporter {
	return &FakeReporter{
		SpecWillRunSummaries: make([]*types.SpecSummary, 0),
		SpecSummaries:        make([]*types.SpecSummary, 0),
	}
}

func (fakeR *FakeReporter) SpecSuiteWillBegin(config config.GinkgoConfigType, summary *types.SuiteSummary) {
	fakeR.Config = config
	fakeR.BeginSummary = summary
}

func (fakeR *FakeReporter) BeforeSuiteDidRun(setupSummary *types.SetupSummary) {
	fakeR.BeforeSuiteSummary = setupSummary
}

func (fakeR *FakeReporter) SpecWillRun(specSummary *types.SpecSummary) {
	if fakeR.SpecWillRunStub != nil {
		fakeR.SpecWillRunStub(specSummary)
	}
	fakeR.SpecWillRunSummaries = append(fakeR.SpecWillRunSummaries, specSummary)
}

func (fakeR *FakeReporter) SpecDidComplete(specSummary *types.SpecSummary) {
	if fakeR.SpecDidCompleteStub != nil {
		fakeR.SpecDidCompleteStub(specSummary)
	}
	fakeR.SpecSummaries = append(fakeR.SpecSummaries, specSummary)
}

func (fakeR *FakeReporter) AfterSuiteDidRun(setupSummary *types.SetupSummary) {
	fakeR.AfterSuiteSummary = setupSummary
}

func (fakeR *FakeReporter) SpecSuiteDidEnd(summary *types.SuiteSummary) {
	fakeR.EndSummary = summary
}
