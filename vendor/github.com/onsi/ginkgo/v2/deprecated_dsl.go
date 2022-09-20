package ginkgo

import (
	"time"

	"github.com/onsi/ginkgo/v2/internal"
	"github.com/onsi/ginkgo/v2/internal/global"
	"github.com/onsi/ginkgo/v2/reporters"
	"github.com/onsi/ginkgo/v2/types"
)

/*
Deprecated: Done Channel for asynchronous testing

The Done channel pattern is no longer supported in Ginkgo 2.0.
See here for better patterns for asynchronous testing: https://onsi.github.io/ginkgo/#patterns-for-asynchronous-testing

For a migration guide see: https://onsi.github.io/ginkgo/MIGRATING_TO_V2#removed-async-testing
*/
type Done = internal.Done

/*
Deprecated: Custom Ginkgo test reporters are deprecated in Ginkgo 2.0.

Use Ginkgo's reporting nodes instead and 2.0 reporting infrastructure instead. You can learn more here: https://onsi.github.io/ginkgo/#reporting-infrastructure
For a migration guide see: https://onsi.github.io/ginkgo/MIGRATING_TO_V2#removed-custom-reporters
*/
type Reporter = reporters.DeprecatedReporter

/*
Deprecated: Custom Reporters have been removed in Ginkgo 2.0.  RunSpecsWithDefaultAndCustomReporters will simply call RunSpecs()

Use Ginkgo's reporting nodes instead and 2.0 reporting infrastructure instead. You can learn more here: https://onsi.github.io/ginkgo/#reporting-infrastructure
For a migration guide see: https://onsi.github.io/ginkgo/MIGRATING_TO_V2#removed-custom-reporters
*/
func RunSpecsWithDefaultAndCustomReporters(t GinkgoTestingT, description string, _ []Reporter) bool {
	deprecationTracker.TrackDeprecation(types.Deprecations.CustomReporter())
	return RunSpecs(t, description)
}

/*
Deprecated: Custom Reporters have been removed in Ginkgo 2.0.  RunSpecsWithCustomReporters will simply call RunSpecs()

Use Ginkgo's reporting nodes instead and 2.0 reporting infrastructure instead. You can learn more here: https://onsi.github.io/ginkgo/#reporting-infrastructure
For a migration guide see: https://onsi.github.io/ginkgo/MIGRATING_TO_V2#removed-custom-reporters
*/
func RunSpecsWithCustomReporters(t GinkgoTestingT, description string, _ []Reporter) bool {
	deprecationTracker.TrackDeprecation(types.Deprecations.CustomReporter())
	return RunSpecs(t, description)
}

/*
Deprecated: GinkgoTestDescription has been replaced with SpecReport.

Use CurrentSpecReport() instead.
You can learn more here: https://onsi.github.io/ginkgo/#getting-a-report-for-the-current-spec
The SpecReport type is documented here: https://pkg.go.dev/github.com/onsi/ginkgo/v2/types#SpecReport
*/
type DeprecatedGinkgoTestDescription struct {
	FullTestText   string
	ComponentTexts []string
	TestText       string

	FileName   string
	LineNumber int

	Failed   bool
	Duration time.Duration
}
type GinkgoTestDescription = DeprecatedGinkgoTestDescription

/*
Deprecated: CurrentGinkgoTestDescription has been replaced with CurrentSpecReport.

Use CurrentSpecReport() instead.
You can learn more here: https://onsi.github.io/ginkgo/#getting-a-report-for-the-current-spec
The SpecReport type is documented here: https://pkg.go.dev/github.com/onsi/ginkgo/v2/types#SpecReport
*/
func CurrentGinkgoTestDescription() DeprecatedGinkgoTestDescription {
	deprecationTracker.TrackDeprecation(
		types.Deprecations.CurrentGinkgoTestDescription(),
		types.NewCodeLocation(1),
	)
	report := global.Suite.CurrentSpecReport()
	if report.State == types.SpecStateInvalid {
		return GinkgoTestDescription{}
	}
	componentTexts := []string{}
	componentTexts = append(componentTexts, report.ContainerHierarchyTexts...)
	componentTexts = append(componentTexts, report.LeafNodeText)

	return DeprecatedGinkgoTestDescription{
		ComponentTexts: componentTexts,
		FullTestText:   report.FullText(),
		TestText:       report.LeafNodeText,
		FileName:       report.LeafNodeLocation.FileName,
		LineNumber:     report.LeafNodeLocation.LineNumber,
		Failed:         report.State.Is(types.SpecStateFailureStates),
		Duration:       report.RunTime,
	}
}

/*
Deprecated: GinkgoParallelNode() has been renamed to GinkgoParallelProcess()
*/
func GinkgoParallelNode() int {
	deprecationTracker.TrackDeprecation(
		types.Deprecations.ParallelNode(),
		types.NewCodeLocation(1),
	)
	return GinkgoParallelProcess()
}

/*
Deprecated: Benchmarker has been removed from Ginkgo 2.0

Use Gomega's gmeasure package instead.
You can learn more here: https://onsi.github.io/ginkgo/#benchmarking-code
*/
type Benchmarker interface {
	Time(name string, body func(), info ...interface{}) (elapsedTime time.Duration)
	RecordValue(name string, value float64, info ...interface{})
	RecordValueWithPrecision(name string, value float64, units string, precision int, info ...interface{})
}

/*
Deprecated: Measure() has been removed from Ginkgo 2.0

Use Gomega's gmeasure package instead.
You can learn more here: https://onsi.github.io/ginkgo/#benchmarking-code
*/
func Measure(_ ...interface{}) bool {
	deprecationTracker.TrackDeprecation(types.Deprecations.Measure(), types.NewCodeLocation(1))
	return true
}
