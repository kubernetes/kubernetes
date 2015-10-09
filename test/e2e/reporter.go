package e2e

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/reporters/stenographer"
	"github.com/onsi/ginkgo/types"
)

// maxDescriptionLength define the maximum number of characters showed for the
// spec description in console. After this number, the description will be
// trimmed and suffixed by ' ...'
const maxDescriptionLength = 100

// TrimCodeLocationOn trims the code location (file path) to left after last
// occurence of this string. This helps copy&paste of broken test files and also
// improves readability
var TrimCodeLocationOn = "/k8s.io/kubernetes/"

type SimpleReporter struct {
	stenographer stenographer.Stenographer

	// Output defines a writer where to send the test output (eg. os.Stdout)
	Output io.Writer
}

// NewSimpleReporter initializes the simple Ginkgo reporter
func NewSimpleReporter() *SimpleReporter {
	return &SimpleReporter{
		Output:       os.Stdout,
		stenographer: stenographer.New(!config.DefaultReporterConfig.NoColor),
	}
}

func (r *SimpleReporter) SpecSuiteWillBegin(config config.GinkgoConfigType, summary *types.SuiteSummary) {
	fmt.Fprintf(r.Output, "=== SUITE %s (%d total specs, %d will run):\n",
		summary.SuiteDescription, summary.NumberOfTotalSpecs, summary.NumberOfSpecsThatWillBeRun)
}

func (r *SimpleReporter) BeforeSuiteDidRun(*types.SetupSummary) {
}

func (r *SimpleReporter) SpecWillRun(spec *types.SpecSummary) {
	r.printRunLine(spec)
}

func (r *SimpleReporter) SpecDidComplete(spec *types.SpecSummary) {
	r.handleSpecFailure(spec)
	r.printStatusLine(spec)
}

func (r *SimpleReporter) AfterSuiteDidRun(setupSummary *types.SetupSummary) {
}

func (r *SimpleReporter) SpecSuiteDidEnd(summary *types.SuiteSummary) {
}

func (r *SimpleReporter) handleSpecFailure(spec *types.SpecSummary) {
	switch spec.State {
	case types.SpecStateFailed:
		r.stenographer.AnnounceSpecFailed(spec, true, false)
	case types.SpecStatePanicked:
		r.stenographer.AnnounceSpecPanicked(spec, true, false)
	case types.SpecStateTimedOut:
		r.stenographer.AnnounceSpecTimedOut(spec, true, false)
	}
}

func (r *SimpleReporter) printStatusLine(spec *types.SpecSummary) {
	runTime := ""
	if runTime = fmt.Sprintf(" (%v)", spec.RunTime); runTime == " (0)" {
		runTime = ""
	}
	fmt.Fprintf(r.Output, "%4s%-16s %s%s\n", " ", convertGinkgoState(spec.State), specDescription(spec), runTime)
}

func (r *SimpleReporter) printRunLine(spec *types.SpecSummary) {
	fmt.Fprintf(r.Output, "=== RUN %s:\n", trimCodeLocation(spec.ComponentCodeLocations[1]))
}

func specDescription(spec *types.SpecSummary) string {
	name := ""
	for _, t := range spec.ComponentTexts[1:len(spec.ComponentTexts)] {
		name += strings.TrimSpace(t) + " "
	}
	if len(name) == 0 {
		name = fmt.Sprintf("FIXME: Invalid name for %q", spec.ComponentTexts)
	}
	return shorten(strings.TrimSpace(name))
}

func shorten(s string) string {
	runes := []rune(s)
	if len(runes) > maxDescriptionLength {
		return string(runes[:maxDescriptionLength]) + " ..."
	}
	return s
}

func bold(v string) string {
	return "\033[1m" + v + "\033[0m"
}

func red(v string) string {
	return "\033[31m" + v + "\033[0m"
}

func magenta(v string) string {
	return "\033[35m" + v + "\033[0m"
}

func convertGinkgoState(s types.SpecState) string {
	switch s {
	case types.SpecStatePassed:
		return bold("ok")
	case types.SpecStateSkipped:
		return magenta("skip")
	case types.SpecStateFailed:
		return red("fail")
	case types.SpecStateTimedOut:
		return red("timed")
	case types.SpecStatePanicked:
		return red("panic")
	case types.SpecStatePending:
		return magenta("pending")
	default:
		return bold(fmt.Sprintf("%v", s))
	}
}

func trimCodeLocation(l types.CodeLocation) string {
	return fmt.Sprintf("%q", l.FileName[strings.LastIndex(l.FileName, TrimCodeLocationOn)+len(TrimCodeLocationOn):])
}
