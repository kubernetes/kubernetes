/*
Ginkgo's Default Reporter

A number of command line flags are available to tweak Ginkgo's default output.

These are documented [here](http://onsi.github.io/ginkgo/#running_tests)
*/
package reporters

import (
	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/reporters/stenographer"
	"github.com/onsi/ginkgo/types"
)

type DefaultReporter struct {
	config        config.DefaultReporterConfigType
	stenographer  stenographer.Stenographer
	specSummaries []*types.SpecSummary
}

func NewDefaultReporter(config config.DefaultReporterConfigType, stenographer stenographer.Stenographer) *DefaultReporter {
	return &DefaultReporter{
		config:       config,
		stenographer: stenographer,
	}
}

func (reporter *DefaultReporter) SpecSuiteWillBegin(config config.GinkgoConfigType, summary *types.SuiteSummary) {
	reporter.stenographer.AnnounceSuite(summary.SuiteDescription, config.RandomSeed, config.RandomizeAllSpecs, reporter.config.Succinct)
	if config.ParallelTotal > 1 {
		reporter.stenographer.AnnounceParallelRun(config.ParallelNode, config.ParallelTotal, summary.NumberOfTotalSpecs, summary.NumberOfSpecsBeforeParallelization, reporter.config.Succinct)
	}
	reporter.stenographer.AnnounceNumberOfSpecs(summary.NumberOfSpecsThatWillBeRun, summary.NumberOfTotalSpecs, reporter.config.Succinct)
}

func (reporter *DefaultReporter) BeforeSuiteDidRun(setupSummary *types.SetupSummary) {
	if setupSummary.State != types.SpecStatePassed {
		reporter.stenographer.AnnounceBeforeSuiteFailure(setupSummary, reporter.config.Succinct, reporter.config.FullTrace)
	}
}

func (reporter *DefaultReporter) AfterSuiteDidRun(setupSummary *types.SetupSummary) {
	if setupSummary.State != types.SpecStatePassed {
		reporter.stenographer.AnnounceAfterSuiteFailure(setupSummary, reporter.config.Succinct, reporter.config.FullTrace)
	}
}

func (reporter *DefaultReporter) SpecWillRun(specSummary *types.SpecSummary) {
	if reporter.config.Verbose && !reporter.config.Succinct && specSummary.State != types.SpecStatePending && specSummary.State != types.SpecStateSkipped {
		reporter.stenographer.AnnounceSpecWillRun(specSummary)
	}
}

func (reporter *DefaultReporter) SpecDidComplete(specSummary *types.SpecSummary) {
	switch specSummary.State {
	case types.SpecStatePassed:
		if specSummary.IsMeasurement {
			reporter.stenographer.AnnounceSuccesfulMeasurement(specSummary, reporter.config.Succinct)
		} else if specSummary.RunTime.Seconds() >= reporter.config.SlowSpecThreshold {
			reporter.stenographer.AnnounceSuccesfulSlowSpec(specSummary, reporter.config.Succinct)
		} else {
			reporter.stenographer.AnnounceSuccesfulSpec(specSummary)
		}
	case types.SpecStatePending:
		reporter.stenographer.AnnouncePendingSpec(specSummary, reporter.config.NoisyPendings && !reporter.config.Succinct)
	case types.SpecStateSkipped:
		reporter.stenographer.AnnounceSkippedSpec(specSummary, reporter.config.Succinct, reporter.config.FullTrace)
	case types.SpecStateTimedOut:
		reporter.stenographer.AnnounceSpecTimedOut(specSummary, reporter.config.Succinct, reporter.config.FullTrace)
	case types.SpecStatePanicked:
		reporter.stenographer.AnnounceSpecPanicked(specSummary, reporter.config.Succinct, reporter.config.FullTrace)
	case types.SpecStateFailed:
		reporter.stenographer.AnnounceSpecFailed(specSummary, reporter.config.Succinct, reporter.config.FullTrace)
	}

	reporter.specSummaries = append(reporter.specSummaries, specSummary)
}

func (reporter *DefaultReporter) SpecSuiteDidEnd(summary *types.SuiteSummary) {
	reporter.stenographer.SummarizeFailures(reporter.specSummaries)
	reporter.stenographer.AnnounceSpecRunCompletion(summary, reporter.config.Succinct)
}
