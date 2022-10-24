/*
Ginkgo's Default Reporter

A number of command line flags are available to tweak Ginkgo's default output.

These are documented [here](http://onsi.github.io/ginkgo/#running_tests)
*/
package reporters

import (
	"fmt"
	"io"
	"runtime"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2/formatter"
	"github.com/onsi/ginkgo/v2/types"
)

type DefaultReporter struct {
	conf   types.ReporterConfig
	writer io.Writer

	// managing the emission stream
	lastChar                 string
	lastEmissionWasDelimiter bool

	// rendering
	specDenoter  string
	retryDenoter string
	formatter    formatter.Formatter
}

func NewDefaultReporterUnderTest(conf types.ReporterConfig, writer io.Writer) *DefaultReporter {
	reporter := NewDefaultReporter(conf, writer)
	reporter.formatter = formatter.New(formatter.ColorModePassthrough)

	return reporter
}

func NewDefaultReporter(conf types.ReporterConfig, writer io.Writer) *DefaultReporter {
	reporter := &DefaultReporter{
		conf:   conf,
		writer: writer,

		lastChar:                 "\n",
		lastEmissionWasDelimiter: false,

		specDenoter:  "•",
		retryDenoter: "↺",
		formatter:    formatter.NewWithNoColorBool(conf.NoColor),
	}
	if runtime.GOOS == "windows" {
		reporter.specDenoter = "+"
		reporter.retryDenoter = "R"
	}

	return reporter
}

/* The Reporter Interface */

func (r *DefaultReporter) SuiteWillBegin(report types.Report) {
	if r.conf.Verbosity().Is(types.VerbosityLevelSuccinct) {
		r.emit(r.f("[%d] {{bold}}%s{{/}} ", report.SuiteConfig.RandomSeed, report.SuiteDescription))
		if len(report.SuiteLabels) > 0 {
			r.emit(r.f("{{coral}}[%s]{{/}} ", strings.Join(report.SuiteLabels, ", ")))
		}
		r.emit(r.f("- %d/%d specs ", report.PreRunStats.SpecsThatWillRun, report.PreRunStats.TotalSpecs))
		if report.SuiteConfig.ParallelTotal > 1 {
			r.emit(r.f("- %d procs ", report.SuiteConfig.ParallelTotal))
		}
	} else {
		banner := r.f("Running Suite: %s - %s", report.SuiteDescription, report.SuitePath)
		r.emitBlock(banner)
		bannerWidth := len(banner)
		if len(report.SuiteLabels) > 0 {
			labels := strings.Join(report.SuiteLabels, ", ")
			r.emitBlock(r.f("{{coral}}[%s]{{/}} ", labels))
			if len(labels)+2 > bannerWidth {
				bannerWidth = len(labels) + 2
			}
		}
		r.emitBlock(strings.Repeat("=", bannerWidth))

		out := r.f("Random Seed: {{bold}}%d{{/}}", report.SuiteConfig.RandomSeed)
		if report.SuiteConfig.RandomizeAllSpecs {
			out += r.f(" - will randomize all specs")
		}
		r.emitBlock(out)
		r.emit("\n")
		r.emitBlock(r.f("Will run {{bold}}%d{{/}} of {{bold}}%d{{/}} specs", report.PreRunStats.SpecsThatWillRun, report.PreRunStats.TotalSpecs))
		if report.SuiteConfig.ParallelTotal > 1 {
			r.emitBlock(r.f("Running in parallel across {{bold}}%d{{/}} processes", report.SuiteConfig.ParallelTotal))
		}
	}
}

func (r *DefaultReporter) WillRun(report types.SpecReport) {
	if r.conf.Verbosity().LT(types.VerbosityLevelVerbose) || report.State.Is(types.SpecStatePending|types.SpecStateSkipped) {
		return
	}

	r.emitDelimiter()
	indentation := uint(0)
	if report.LeafNodeType.Is(types.NodeTypesForSuiteLevelNodes) {
		r.emitBlock(r.f("{{bold}}[%s] %s{{/}}", report.LeafNodeType.String(), report.LeafNodeText))
	} else {
		if len(report.ContainerHierarchyTexts) > 0 {
			r.emitBlock(r.cycleJoin(report.ContainerHierarchyTexts, " "))
			indentation = 1
		}
		line := r.fi(indentation, "{{bold}}%s{{/}}", report.LeafNodeText)
		labels := report.Labels()
		if len(labels) > 0 {
			line += r.f(" {{coral}}[%s]{{/}}", strings.Join(labels, ", "))
		}
		r.emitBlock(line)
	}
	r.emitBlock(r.fi(indentation, "{{gray}}%s{{/}}", report.LeafNodeLocation))
}

func (r *DefaultReporter) DidRun(report types.SpecReport) {
	v := r.conf.Verbosity()
	var header, highlightColor string
	includeRuntime, emitGinkgoWriterOutput, stream, denoter := true, true, false, r.specDenoter
	succinctLocationBlock := v.Is(types.VerbosityLevelSuccinct)

	hasGW := report.CapturedGinkgoWriterOutput != ""
	hasStd := report.CapturedStdOutErr != ""
	hasEmittableReports := report.ReportEntries.HasVisibility(types.ReportEntryVisibilityAlways) || (report.ReportEntries.HasVisibility(types.ReportEntryVisibilityFailureOrVerbose) && (!report.Failure.IsZero() || v.GTE(types.VerbosityLevelVerbose)))

	if report.LeafNodeType.Is(types.NodeTypesForSuiteLevelNodes) {
		denoter = fmt.Sprintf("[%s]", report.LeafNodeType)
	}

	highlightColor = r.highlightColorForState(report.State)

	switch report.State {
	case types.SpecStatePassed:
		succinctLocationBlock = v.LT(types.VerbosityLevelVerbose)
		emitGinkgoWriterOutput = (r.conf.AlwaysEmitGinkgoWriter || v.GTE(types.VerbosityLevelVerbose)) && hasGW
		if report.LeafNodeType.Is(types.NodeTypesForSuiteLevelNodes) {
			if v.GTE(types.VerbosityLevelVerbose) || hasStd || hasEmittableReports {
				header = fmt.Sprintf("%s PASSED", denoter)
			} else {
				return
			}
		} else {
			header, stream = denoter, true
			if report.NumAttempts > 1 && report.MaxFlakeAttempts > 1 {
				header, stream = fmt.Sprintf("%s [FLAKEY TEST - TOOK %d ATTEMPTS TO PASS]", r.retryDenoter, report.NumAttempts), false
			}
			if report.RunTime > r.conf.SlowSpecThreshold {
				header, stream = fmt.Sprintf("%s [SLOW TEST]", header), false
			}
		}
		if hasStd || emitGinkgoWriterOutput || hasEmittableReports {
			stream = false
		}
	case types.SpecStatePending:
		includeRuntime, emitGinkgoWriterOutput = false, false
		if v.Is(types.VerbosityLevelSuccinct) {
			header, stream = "P", true
		} else {
			header, succinctLocationBlock = "P [PENDING]", v.LT(types.VerbosityLevelVeryVerbose)
		}
	case types.SpecStateSkipped:
		if report.Failure.Message != "" || v.Is(types.VerbosityLevelVeryVerbose) {
			header = "S [SKIPPED]"
		} else {
			header, stream = "S", true
		}
	case types.SpecStateFailed:
		header = fmt.Sprintf("%s [FAILED]", denoter)
	case types.SpecStateTimedout:
		header = fmt.Sprintf("%s [TIMEDOUT]", denoter)
	case types.SpecStatePanicked:
		header = fmt.Sprintf("%s! [PANICKED]", denoter)
	case types.SpecStateInterrupted:
		header = fmt.Sprintf("%s! [INTERRUPTED]", denoter)
	case types.SpecStateAborted:
		header = fmt.Sprintf("%s! [ABORTED]", denoter)
	}

	if report.State.Is(types.SpecStateFailureStates) && report.MaxMustPassRepeatedly > 1 {
		header, stream = fmt.Sprintf("%s DURING REPETITION #%d", header, report.NumAttempts), false
	}
	// Emit stream and return
	if stream {
		r.emit(r.f(highlightColor + header + "{{/}}"))
		return
	}

	// Emit header
	r.emitDelimiter()
	if includeRuntime {
		header = r.f("%s [%.3f seconds]", header, report.RunTime.Seconds())
	}
	r.emitBlock(r.f(highlightColor + header + "{{/}}"))

	// Emit Code Location Block
	r.emitBlock(r.codeLocationBlock(report, highlightColor, succinctLocationBlock, false))

	//Emit Stdout/Stderr Output
	if hasStd {
		r.emitBlock("\n")
		r.emitBlock(r.fi(1, "{{gray}}Begin Captured StdOut/StdErr Output >>{{/}}"))
		r.emitBlock(r.fi(2, "%s", report.CapturedStdOutErr))
		r.emitBlock(r.fi(1, "{{gray}}<< End Captured StdOut/StdErr Output{{/}}"))
	}

	//Emit Captured GinkgoWriter Output
	if emitGinkgoWriterOutput && hasGW {
		r.emitBlock("\n")
		r.emitGinkgoWriterOutput(1, report.CapturedGinkgoWriterOutput, 0)
	}

	if hasEmittableReports {
		r.emitBlock("\n")
		r.emitBlock(r.fi(1, "{{gray}}Begin Report Entries >>{{/}}"))
		reportEntries := report.ReportEntries.WithVisibility(types.ReportEntryVisibilityAlways)
		if !report.Failure.IsZero() || v.GTE(types.VerbosityLevelVerbose) {
			reportEntries = report.ReportEntries.WithVisibility(types.ReportEntryVisibilityAlways, types.ReportEntryVisibilityFailureOrVerbose)
		}
		for _, entry := range reportEntries {
			r.emitBlock(r.fi(2, "{{bold}}"+entry.Name+"{{gray}} - %s @ %s{{/}}", entry.Location, entry.Time.Format(types.GINKGO_TIME_FORMAT)))
			if representation := entry.StringRepresentation(); representation != "" {
				r.emitBlock(r.fi(3, representation))
			}
		}
		r.emitBlock(r.fi(1, "{{gray}}<< End Report Entries{{/}}"))
	}

	// Emit Failure Message
	if !report.Failure.IsZero() {
		r.emitBlock("\n")
		r.EmitFailure(1, report.State, report.Failure, false)
	}

	if len(report.AdditionalFailures) > 0 {
		if v.GTE(types.VerbosityLevelVerbose) {
			r.emitBlock("\n")
			r.emitBlock(r.fi(1, "{{bold}}There were additional failures detected after the initial failure:{{/}}"))
			for i, additionalFailure := range report.AdditionalFailures {
				r.EmitFailure(2, additionalFailure.State, additionalFailure.Failure, true)
				if i < len(report.AdditionalFailures)-1 {
					r.emitBlock(r.fi(2, "{{gray}}%s{{/}}", strings.Repeat("-", 10)))
				}
			}
		} else {
			r.emitBlock("\n")
			r.emitBlock(r.fi(1, "{{bold}}There were additional failures detected after the initial failure.  Here's a summary - for full details run Ginkgo in verbose mode:{{/}}"))
			for _, additionalFailure := range report.AdditionalFailures {
				r.emitBlock(r.fi(2, r.highlightColorForState(additionalFailure.State)+"[%s]{{/}} in [%s] at %s",
					r.humanReadableState(additionalFailure.State),
					additionalFailure.Failure.FailureNodeType,
					additionalFailure.Failure.Location,
				))
			}

		}
	}

	r.emitDelimiter()
}

func (r *DefaultReporter) highlightColorForState(state types.SpecState) string {
	switch state {
	case types.SpecStatePassed:
		return "{{green}}"
	case types.SpecStatePending:
		return "{{yellow}}"
	case types.SpecStateSkipped:
		return "{{cyan}}"
	case types.SpecStateFailed:
		return "{{red}}"
	case types.SpecStateTimedout:
		return "{{orange}}"
	case types.SpecStatePanicked:
		return "{{magenta}}"
	case types.SpecStateInterrupted:
		return "{{orange}}"
	case types.SpecStateAborted:
		return "{{coral}}"
	default:
		return "{{gray}}"
	}
}

func (r *DefaultReporter) humanReadableState(state types.SpecState) string {
	return strings.ToUpper(state.String())
}

func (r *DefaultReporter) EmitFailure(indent uint, state types.SpecState, failure types.Failure, includeState bool) {
	highlightColor := r.highlightColorForState(state)
	if includeState {
		r.emitBlock(r.fi(indent, highlightColor+"[%s]{{/}}", r.humanReadableState(state)))
	}
	r.emitBlock(r.fi(indent, highlightColor+"%s{{/}}", failure.Message))
	r.emitBlock(r.fi(indent, highlightColor+"In {{bold}}[%s]{{/}}"+highlightColor+" at: {{bold}}%s{{/}}\n", failure.FailureNodeType, failure.Location))
	if failure.ForwardedPanic != "" {
		r.emitBlock("\n")
		r.emitBlock(r.fi(indent, highlightColor+"%s{{/}}", failure.ForwardedPanic))
	}

	if r.conf.FullTrace || failure.ForwardedPanic != "" {
		r.emitBlock("\n")
		r.emitBlock(r.fi(indent, highlightColor+"Full Stack Trace{{/}}"))
		r.emitBlock(r.fi(indent+1, "%s", failure.Location.FullStackTrace))
	}

	if !failure.ProgressReport.IsZero() {
		r.emitBlock("\n")
		r.emitProgressReport(indent, false, failure.ProgressReport)
	}
}

func (r *DefaultReporter) SuiteDidEnd(report types.Report) {
	failures := report.SpecReports.WithState(types.SpecStateFailureStates)
	if len(failures) > 0 {
		r.emitBlock("\n\n")
		if len(failures) > 1 {
			r.emitBlock(r.f("{{red}}{{bold}}Summarizing %d Failures:{{/}}", len(failures)))
		} else {
			r.emitBlock(r.f("{{red}}{{bold}}Summarizing 1 Failure:{{/}}"))
		}
		for _, specReport := range failures {
			highlightColor, heading := "{{red}}", "[FAIL]"
			switch specReport.State {
			case types.SpecStatePanicked:
				highlightColor, heading = "{{magenta}}", "[PANICKED!]"
			case types.SpecStateAborted:
				highlightColor, heading = "{{coral}}", "[ABORTED]"
			case types.SpecStateTimedout:
				highlightColor, heading = "{{orange}}", "[TIMEDOUT]"
			case types.SpecStateInterrupted:
				highlightColor, heading = "{{orange}}", "[INTERRUPTED]"
			}
			locationBlock := r.codeLocationBlock(specReport, highlightColor, true, true)
			r.emitBlock(r.fi(1, highlightColor+"%s{{/}} %s", heading, locationBlock))
		}
	}

	//summarize the suite
	if r.conf.Verbosity().Is(types.VerbosityLevelSuccinct) && report.SuiteSucceeded {
		r.emit(r.f(" {{green}}SUCCESS!{{/}} %s ", report.RunTime))
		return
	}

	r.emitBlock("\n")
	color, status := "{{green}}{{bold}}", "SUCCESS!"
	if !report.SuiteSucceeded {
		color, status = "{{red}}{{bold}}", "FAIL!"
	}

	specs := report.SpecReports.WithLeafNodeType(types.NodeTypeIt) //exclude any suite setup nodes
	r.emitBlock(r.f(color+"Ran %d of %d Specs in %.3f seconds{{/}}",
		specs.CountWithState(types.SpecStatePassed)+specs.CountWithState(types.SpecStateFailureStates),
		report.PreRunStats.TotalSpecs,
		report.RunTime.Seconds()),
	)

	switch len(report.SpecialSuiteFailureReasons) {
	case 0:
		r.emit(r.f(color+"%s{{/}} -- ", status))
	case 1:
		r.emit(r.f(color+"%s - %s{{/}} -- ", status, report.SpecialSuiteFailureReasons[0]))
	default:
		r.emitBlock(r.f(color+"%s - %s{{/}}\n", status, strings.Join(report.SpecialSuiteFailureReasons, ", ")))
	}

	if len(specs) == 0 && report.SpecReports.WithLeafNodeType(types.NodeTypeBeforeSuite|types.NodeTypeSynchronizedBeforeSuite).CountWithState(types.SpecStateFailureStates) > 0 {
		r.emit(r.f("{{cyan}}{{bold}}A BeforeSuite node failed so all tests were skipped.{{/}}\n"))
	} else {
		r.emit(r.f("{{green}}{{bold}}%d Passed{{/}} | ", specs.CountWithState(types.SpecStatePassed)))
		r.emit(r.f("{{red}}{{bold}}%d Failed{{/}} | ", specs.CountWithState(types.SpecStateFailureStates)))
		if specs.CountOfFlakedSpecs() > 0 {
			r.emit(r.f("{{light-yellow}}{{bold}}%d Flaked{{/}} | ", specs.CountOfFlakedSpecs()))
		}
		if specs.CountOfRepeatedSpecs() > 0 {
			r.emit(r.f("{{light-yellow}}{{bold}}%d Repeated{{/}} | ", specs.CountOfRepeatedSpecs()))
		}
		r.emit(r.f("{{yellow}}{{bold}}%d Pending{{/}} | ", specs.CountWithState(types.SpecStatePending)))
		r.emit(r.f("{{cyan}}{{bold}}%d Skipped{{/}}\n", specs.CountWithState(types.SpecStateSkipped)))
	}
}

func (r *DefaultReporter) EmitProgressReport(report types.ProgressReport) {
	r.emitDelimiter()

	if report.RunningInParallel {
		r.emit(r.f("{{coral}}Progress Report for Ginkgo Process #{{bold}}%d{{/}}\n", report.ParallelProcess))
	}
	r.emitProgressReport(0, true, report)
	r.emitDelimiter()
}

func (r *DefaultReporter) emitProgressReport(indent uint, emitGinkgoWriterOutput bool, report types.ProgressReport) {
	if report.Message != "" {
		r.emitBlock(r.fi(indent, report.Message+"\n"))
		indent += 1
	}
	if report.LeafNodeText != "" {
		subjectIndent := indent
		if len(report.ContainerHierarchyTexts) > 0 {
			r.emit(r.fi(indent, r.cycleJoin(report.ContainerHierarchyTexts, " ")))
			r.emit(" ")
			subjectIndent = 0
		}
		r.emit(r.fi(subjectIndent, "{{bold}}{{orange}}%s{{/}} (Spec Runtime: %s)\n", report.LeafNodeText, report.Time.Sub(report.SpecStartTime).Round(time.Millisecond)))
		r.emit(r.fi(indent+1, "{{gray}}%s{{/}}\n", report.LeafNodeLocation))
		indent += 1
	}
	if report.CurrentNodeType != types.NodeTypeInvalid {
		r.emit(r.fi(indent, "In {{bold}}{{orange}}[%s]{{/}}", report.CurrentNodeType))
		if report.CurrentNodeText != "" && !report.CurrentNodeType.Is(types.NodeTypeIt) {
			r.emit(r.f(" {{bold}}{{orange}}%s{{/}}", report.CurrentNodeText))
		}

		r.emit(r.f(" (Node Runtime: %s)\n", report.Time.Sub(report.CurrentNodeStartTime).Round(time.Millisecond)))
		r.emit(r.fi(indent+1, "{{gray}}%s{{/}}\n", report.CurrentNodeLocation))
		indent += 1
	}
	if report.CurrentStepText != "" {
		r.emit(r.fi(indent, "At {{bold}}{{orange}}[By Step] %s{{/}} (Step Runtime: %s)\n", report.CurrentStepText, report.Time.Sub(report.CurrentStepStartTime).Round(time.Millisecond)))
		r.emit(r.fi(indent+1, "{{gray}}%s{{/}}\n", report.CurrentStepLocation))
		indent += 1
	}

	if indent > 0 {
		indent -= 1
	}

	if emitGinkgoWriterOutput && report.CapturedGinkgoWriterOutput != "" && (report.RunningInParallel || r.conf.Verbosity().LT(types.VerbosityLevelVerbose)) {
		r.emit("\n")
		r.emitGinkgoWriterOutput(indent, report.CapturedGinkgoWriterOutput, 10)
	}

	if !report.SpecGoroutine().IsZero() {
		r.emit("\n")
		r.emit(r.fi(indent, "{{bold}}{{underline}}Spec Goroutine{{/}}\n"))
		r.emitGoroutines(indent, report.SpecGoroutine())
	}

	if len(report.AdditionalReports) > 0 {
		r.emit("\n")
		r.emitBlock(r.fi(indent, "{{gray}}Begin Additional Progress Reports >>{{/}}"))
		for i, additionalReport := range report.AdditionalReports {
			r.emit(r.fi(indent+1, additionalReport))
			if i < len(report.AdditionalReports)-1 {
				r.emitBlock(r.fi(indent+1, "{{gray}}%s{{/}}", strings.Repeat("-", 10)))
			}
		}
		r.emitBlock(r.fi(indent, "{{gray}}<< End Additional Progress Reports{{/}}"))
	}

	highlightedGoroutines := report.HighlightedGoroutines()
	if len(highlightedGoroutines) > 0 {
		r.emit("\n")
		r.emit(r.fi(indent, "{{bold}}{{underline}}Goroutines of Interest{{/}}\n"))
		r.emitGoroutines(indent, highlightedGoroutines...)
	}

	otherGoroutines := report.OtherGoroutines()
	if len(otherGoroutines) > 0 {
		r.emit("\n")
		r.emit(r.fi(indent, "{{gray}}{{bold}}{{underline}}Other Goroutines{{/}}\n"))
		r.emitGoroutines(indent, otherGoroutines...)
	}
}

func (r *DefaultReporter) emitGinkgoWriterOutput(indent uint, output string, limit int) {
	r.emitBlock(r.fi(indent, "{{gray}}Begin Captured GinkgoWriter Output >>{{/}}"))
	if limit == 0 {
		r.emitBlock(r.fi(indent+1, "%s", output))
	} else {
		lines := strings.Split(output, "\n")
		if len(lines) <= limit {
			r.emitBlock(r.fi(indent+1, "%s", output))
		} else {
			r.emitBlock(r.fi(indent+1, "{{gray}}...{{/}}"))
			for _, line := range lines[len(lines)-limit-1:] {
				r.emitBlock(r.fi(indent+1, "%s", line))
			}
		}
	}
	r.emitBlock(r.fi(indent, "{{gray}}<< End Captured GinkgoWriter Output{{/}}"))
}

func (r *DefaultReporter) emitGoroutines(indent uint, goroutines ...types.Goroutine) {
	for idx, g := range goroutines {
		color := "{{gray}}"
		if g.HasHighlights() {
			color = "{{orange}}"
		}
		r.emit(r.fi(indent, color+"goroutine %d [%s]{{/}}\n", g.ID, g.State))
		for _, fc := range g.Stack {
			if fc.Highlight {
				r.emit(r.fi(indent, color+"{{bold}}> %s{{/}}\n", fc.Function))
				r.emit(r.fi(indent+2, color+"{{bold}}%s:%d{{/}}\n", fc.Filename, fc.Line))
				r.emitSource(indent+3, fc)
			} else {
				r.emit(r.fi(indent+1, "{{gray}}%s{{/}}\n", fc.Function))
				r.emit(r.fi(indent+2, "{{gray}}%s:%d{{/}}\n", fc.Filename, fc.Line))
			}
		}

		if idx+1 < len(goroutines) {
			r.emit("\n")
		}
	}
}

func (r *DefaultReporter) emitSource(indent uint, fc types.FunctionCall) {
	lines := fc.Source
	if len(lines) == 0 {
		return
	}

	lTrim := 100000
	for _, line := range lines {
		lTrimLine := len(line) - len(strings.TrimLeft(line, " \t"))
		if lTrimLine < lTrim && len(line) > 0 {
			lTrim = lTrimLine
		}
	}
	if lTrim == 100000 {
		lTrim = 0
	}

	for idx, line := range lines {
		if len(line) > lTrim {
			line = line[lTrim:]
		}
		if idx == fc.SourceHighlight {
			r.emit(r.fi(indent, "{{bold}}{{orange}}> %s{{/}}\n", line))
		} else {
			r.emit(r.fi(indent, "| %s\n", line))
		}
	}
}

/* Emitting to the writer */
func (r *DefaultReporter) emit(s string) {
	if len(s) > 0 {
		r.lastChar = s[len(s)-1:]
		r.lastEmissionWasDelimiter = false
		r.writer.Write([]byte(s))
	}
}

func (r *DefaultReporter) emitBlock(s string) {
	if len(s) > 0 {
		if r.lastChar != "\n" {
			r.emit("\n")
		}
		r.emit(s)
		if r.lastChar != "\n" {
			r.emit("\n")
		}
	}
}

func (r *DefaultReporter) emitDelimiter() {
	if r.lastEmissionWasDelimiter {
		return
	}
	r.emitBlock(r.f("{{gray}}%s{{/}}", strings.Repeat("-", 30)))
	r.lastEmissionWasDelimiter = true
}

/* Rendering text */
func (r *DefaultReporter) f(format string, args ...interface{}) string {
	return r.formatter.F(format, args...)
}

func (r *DefaultReporter) fi(indentation uint, format string, args ...interface{}) string {
	return r.formatter.Fi(indentation, format, args...)
}

func (r *DefaultReporter) cycleJoin(elements []string, joiner string) string {
	return r.formatter.CycleJoin(elements, joiner, []string{"{{/}}", "{{gray}}"})
}

func (r *DefaultReporter) codeLocationBlock(report types.SpecReport, highlightColor string, succinct bool, usePreciseFailureLocation bool) string {
	texts, locations, labels := []string{}, []types.CodeLocation{}, [][]string{}
	texts, locations, labels = append(texts, report.ContainerHierarchyTexts...), append(locations, report.ContainerHierarchyLocations...), append(labels, report.ContainerHierarchyLabels...)
	if report.LeafNodeType.Is(types.NodeTypesForSuiteLevelNodes) {
		texts = append(texts, r.f("[%s] %s", report.LeafNodeType, report.LeafNodeText))
	} else {
		texts = append(texts, report.LeafNodeText)
	}
	labels = append(labels, report.LeafNodeLabels)
	locations = append(locations, report.LeafNodeLocation)

	failureLocation := report.Failure.FailureNodeLocation
	if usePreciseFailureLocation {
		failureLocation = report.Failure.Location
	}

	switch report.Failure.FailureNodeContext {
	case types.FailureNodeAtTopLevel:
		texts = append([]string{r.f(highlightColor+"{{bold}}TOP-LEVEL [%s]{{/}}", report.Failure.FailureNodeType)}, texts...)
		locations = append([]types.CodeLocation{failureLocation}, locations...)
		labels = append([][]string{{}}, labels...)
	case types.FailureNodeInContainer:
		i := report.Failure.FailureNodeContainerIndex
		texts[i] = r.f(highlightColor+"{{bold}}%s [%s]{{/}}", texts[i], report.Failure.FailureNodeType)
		locations[i] = failureLocation
	case types.FailureNodeIsLeafNode:
		i := len(texts) - 1
		texts[i] = r.f(highlightColor+"{{bold}}[%s] %s{{/}}", report.LeafNodeType, report.LeafNodeText)
		locations[i] = failureLocation
	}

	out := ""
	if succinct {
		out += r.f("%s", r.cycleJoin(texts, " "))
		flattenedLabels := report.Labels()
		if len(flattenedLabels) > 0 {
			out += r.f(" {{coral}}[%s]{{/}}", strings.Join(flattenedLabels, ", "))
		}
		out += "\n"
		if usePreciseFailureLocation {
			out += r.f("{{gray}}%s{{/}}", failureLocation)
		} else {
			out += r.f("{{gray}}%s{{/}}", locations[len(locations)-1])
		}
	} else {
		for i := range texts {
			out += r.fi(uint(i), "%s", texts[i])
			if len(labels[i]) > 0 {
				out += r.f(" {{coral}}[%s]{{/}}", strings.Join(labels[i], ", "))
			}
			out += "\n"
			out += r.fi(uint(i), "{{gray}}%s{{/}}\n", locations[i])
		}
	}
	return out
}
