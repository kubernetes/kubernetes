package reporters

import (
	"github.com/onsi/ginkgo/v2/config"
	"github.com/onsi/ginkgo/v2/types"
)

// Deprecated: DeprecatedReporter was how Ginkgo V1 provided support for CustomReporters
// this has been removed in V2.
// Please read the documentation at:
// https://onsi.github.io/ginkgo/MIGRATING_TO_V2#removed-custom-reporters
// for Ginkgo's new behavior and for a migration path.
type DeprecatedReporter interface {
	SuiteWillBegin(config config.GinkgoConfigType, summary *types.SuiteSummary)
	BeforeSuiteDidRun(setupSummary *types.SetupSummary)
	SpecWillRun(specSummary *types.SpecSummary)
	SpecDidComplete(specSummary *types.SpecSummary)
	AfterSuiteDidRun(setupSummary *types.SetupSummary)
	SuiteDidEnd(summary *types.SuiteSummary)
}

// ReportViaDeprecatedReporter takes a V1 custom reporter and a V2 report and
// calls the custom reporter's methods with appropriately transformed data from the V2 report.
//
// ReportViaDeprecatedReporter should be called in a `ReportAfterSuite()`
//
// Deprecated: ReportViaDeprecatedReporter method exists to help developer bridge between deprecated V1 functionality and the new
// reporting support in V2.  It will be removed in a future minor version of Ginkgo.
func ReportViaDeprecatedReporter(reporter DeprecatedReporter, report types.Report) {
	conf := config.DeprecatedGinkgoConfigType{
		RandomSeed:        report.SuiteConfig.RandomSeed,
		RandomizeAllSpecs: report.SuiteConfig.RandomizeAllSpecs,
		FocusStrings:      report.SuiteConfig.FocusStrings,
		SkipStrings:       report.SuiteConfig.SkipStrings,
		FailOnPending:     report.SuiteConfig.FailOnPending,
		FailFast:          report.SuiteConfig.FailFast,
		FlakeAttempts:     report.SuiteConfig.FlakeAttempts,
		EmitSpecProgress:  false,
		DryRun:            report.SuiteConfig.DryRun,
		ParallelNode:      report.SuiteConfig.ParallelProcess,
		ParallelTotal:     report.SuiteConfig.ParallelTotal,
		SyncHost:          report.SuiteConfig.ParallelHost,
		StreamHost:        report.SuiteConfig.ParallelHost,
	}

	summary := &types.DeprecatedSuiteSummary{
		SuiteDescription: report.SuiteDescription,
		SuiteID:          report.SuitePath,

		NumberOfSpecsBeforeParallelization: report.PreRunStats.TotalSpecs,
		NumberOfTotalSpecs:                 report.PreRunStats.TotalSpecs,
		NumberOfSpecsThatWillBeRun:         report.PreRunStats.SpecsThatWillRun,
	}

	reporter.SuiteWillBegin(conf, summary)

	for _, spec := range report.SpecReports {
		switch spec.LeafNodeType {
		case types.NodeTypeBeforeSuite, types.NodeTypeSynchronizedBeforeSuite:
			setupSummary := &types.DeprecatedSetupSummary{
				ComponentType:  spec.LeafNodeType,
				CodeLocation:   spec.LeafNodeLocation,
				State:          spec.State,
				RunTime:        spec.RunTime,
				Failure:        failureFor(spec),
				CapturedOutput: spec.CombinedOutput(),
				SuiteID:        report.SuitePath,
			}
			reporter.BeforeSuiteDidRun(setupSummary)
		case types.NodeTypeAfterSuite, types.NodeTypeSynchronizedAfterSuite:
			setupSummary := &types.DeprecatedSetupSummary{
				ComponentType:  spec.LeafNodeType,
				CodeLocation:   spec.LeafNodeLocation,
				State:          spec.State,
				RunTime:        spec.RunTime,
				Failure:        failureFor(spec),
				CapturedOutput: spec.CombinedOutput(),
				SuiteID:        report.SuitePath,
			}
			reporter.AfterSuiteDidRun(setupSummary)
		case types.NodeTypeIt:
			componentTexts, componentCodeLocations := []string{}, []types.CodeLocation{}
			componentTexts = append(componentTexts, spec.ContainerHierarchyTexts...)
			componentCodeLocations = append(componentCodeLocations, spec.ContainerHierarchyLocations...)
			componentTexts = append(componentTexts, spec.LeafNodeText)
			componentCodeLocations = append(componentCodeLocations, spec.LeafNodeLocation)

			specSummary := &types.DeprecatedSpecSummary{
				ComponentTexts:         componentTexts,
				ComponentCodeLocations: componentCodeLocations,
				State:                  spec.State,
				RunTime:                spec.RunTime,
				Failure:                failureFor(spec),
				NumberOfSamples:        spec.NumAttempts,
				CapturedOutput:         spec.CombinedOutput(),
				SuiteID:                report.SuitePath,
			}
			reporter.SpecWillRun(specSummary)
			reporter.SpecDidComplete(specSummary)

			switch spec.State {
			case types.SpecStatePending:
				summary.NumberOfPendingSpecs += 1
			case types.SpecStateSkipped:
				summary.NumberOfSkippedSpecs += 1
			case types.SpecStateFailed, types.SpecStatePanicked, types.SpecStateInterrupted:
				summary.NumberOfFailedSpecs += 1
			case types.SpecStatePassed:
				summary.NumberOfPassedSpecs += 1
				if spec.NumAttempts > 1 {
					summary.NumberOfFlakedSpecs += 1
				}
			}
		}
	}

	summary.SuiteSucceeded = report.SuiteSucceeded
	summary.RunTime = report.RunTime

	reporter.SuiteDidEnd(summary)
}

func failureFor(spec types.SpecReport) types.DeprecatedSpecFailure {
	if spec.Failure.IsZero() {
		return types.DeprecatedSpecFailure{}
	}

	index := 0
	switch spec.Failure.FailureNodeContext {
	case types.FailureNodeInContainer:
		index = spec.Failure.FailureNodeContainerIndex
	case types.FailureNodeAtTopLevel:
		index = -1
	case types.FailureNodeIsLeafNode:
		index = len(spec.ContainerHierarchyTexts) - 1
		if spec.LeafNodeText != "" {
			index += 1
		}
	}

	return types.DeprecatedSpecFailure{
		Message:               spec.Failure.Message,
		Location:              spec.Failure.Location,
		ForwardedPanic:        spec.Failure.ForwardedPanic,
		ComponentIndex:        index,
		ComponentType:         spec.Failure.FailureNodeType,
		ComponentCodeLocation: spec.Failure.FailureNodeLocation,
	}
}
