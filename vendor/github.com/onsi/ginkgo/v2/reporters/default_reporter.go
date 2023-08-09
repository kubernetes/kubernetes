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

	runningInParallel bool
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

func (r *DefaultReporter) SuiteDidEnd(report types.Report) {
	failures := report.SpecReports.WithState(types.SpecStateFailureStates)
	if len(failures) > 0 {
		r.emitBlock("\n")
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
			locationBlock := r.codeLocationBlock(specReport, highlightColor, false, true)
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

func (r *DefaultReporter) WillRun(report types.SpecReport) {
	v := r.conf.Verbosity()
	if v.LT(types.VerbosityLevelVerbose) || report.State.Is(types.SpecStatePending|types.SpecStateSkipped) || report.RunningInParallel {
		return
	}

	r.emitDelimiter(0)
	r.emitBlock(r.f(r.codeLocationBlock(report, "{{/}}", v.Is(types.VerbosityLevelVeryVerbose), false)))
}

func (r *DefaultReporter) DidRun(report types.SpecReport) {
	v := r.conf.Verbosity()
	inParallel := report.RunningInParallel

	header := r.specDenoter
	if report.LeafNodeType.Is(types.NodeTypesForSuiteLevelNodes) {
		header = fmt.Sprintf("[%s]", report.LeafNodeType)
	}
	highlightColor := r.highlightColorForState(report.State)

	// have we already been streaming the timeline?
	timelineHasBeenStreaming := v.GTE(types.VerbosityLevelVerbose) && !inParallel

	// should we show the timeline?
	var timeline types.Timeline
	showTimeline := !timelineHasBeenStreaming && (v.GTE(types.VerbosityLevelVerbose) || report.Failed())
	if showTimeline {
		timeline = report.Timeline().WithoutHiddenReportEntries()
		keepVeryVerboseSpecEvents := v.Is(types.VerbosityLevelVeryVerbose) ||
			(v.Is(types.VerbosityLevelVerbose) && r.conf.ShowNodeEvents) ||
			(report.Failed() && r.conf.ShowNodeEvents)
		if !keepVeryVerboseSpecEvents {
			timeline = timeline.WithoutVeryVerboseSpecEvents()
		}
		if len(timeline) == 0 && report.CapturedGinkgoWriterOutput == "" {
			// the timeline is completely empty - don't show it
			showTimeline = false
		}
		if v.LT(types.VerbosityLevelVeryVerbose) && report.CapturedGinkgoWriterOutput == "" && len(timeline) > 0 {
			//if we aren't -vv and the timeline only has a single failure, don't show it as it will appear at the end of the report
			failure, isFailure := timeline[0].(types.Failure)
			if isFailure && (len(timeline) == 1 || (len(timeline) == 2 && failure.AdditionalFailure != nil)) {
				showTimeline = false
			}
		}
	}

	// should we have a separate section for always-visible reports?
	showSeparateVisibilityAlwaysReportsSection := !timelineHasBeenStreaming && !showTimeline && report.ReportEntries.HasVisibility(types.ReportEntryVisibilityAlways)

	// should we have a separate section for captured stdout/stderr
	showSeparateStdSection := inParallel && (report.CapturedStdOutErr != "")

	// given all that - do we have any actual content to show? or are we a single denoter in a stream?
	reportHasContent := v.Is(types.VerbosityLevelVeryVerbose) || showTimeline || showSeparateVisibilityAlwaysReportsSection || showSeparateStdSection || report.Failed() || (v.Is(types.VerbosityLevelVerbose) && !report.State.Is(types.SpecStateSkipped))

	// should we show a runtime?
	includeRuntime := !report.State.Is(types.SpecStateSkipped|types.SpecStatePending) || (report.State.Is(types.SpecStateSkipped) && report.Failure.Message != "")

	// should we show the codelocation block?
	showCodeLocation := !timelineHasBeenStreaming || !report.State.Is(types.SpecStatePassed)

	switch report.State {
	case types.SpecStatePassed:
		if report.LeafNodeType.Is(types.NodeTypesForSuiteLevelNodes) && !reportHasContent {
			return
		}
		if report.LeafNodeType.Is(types.NodeTypesForSuiteLevelNodes) {
			header = fmt.Sprintf("%s PASSED", header)
		}
		if report.NumAttempts > 1 && report.MaxFlakeAttempts > 1 {
			header, reportHasContent = fmt.Sprintf("%s [FLAKEY TEST - TOOK %d ATTEMPTS TO PASS]", r.retryDenoter, report.NumAttempts), true
		}
	case types.SpecStatePending:
		header = "P"
		if v.GT(types.VerbosityLevelSuccinct) {
			header, reportHasContent = "P [PENDING]", true
		}
	case types.SpecStateSkipped:
		header = "S"
		if v.Is(types.VerbosityLevelVeryVerbose) || (v.Is(types.VerbosityLevelVerbose) && report.Failure.Message != "") {
			header, reportHasContent = "S [SKIPPED]", true
		}
	default:
		header = fmt.Sprintf("%s [%s]", header, r.humanReadableState(report.State))
		if report.MaxMustPassRepeatedly > 1 {
			header = fmt.Sprintf("%s DURING REPETITION #%d", header, report.NumAttempts)
		}
	}

	// If we have no content to show, jsut emit the header and return
	if !reportHasContent {
		r.emit(r.f(highlightColor + header + "{{/}}"))
		return
	}

	if includeRuntime {
		header = r.f("%s [%.3f seconds]", header, report.RunTime.Seconds())
	}

	// Emit header
	if !timelineHasBeenStreaming {
		r.emitDelimiter(0)
	}
	r.emitBlock(r.f(highlightColor + header + "{{/}}"))
	if showCodeLocation {
		r.emitBlock(r.codeLocationBlock(report, highlightColor, v.Is(types.VerbosityLevelVeryVerbose), false))
	}

	//Emit Stdout/Stderr Output
	if showSeparateStdSection {
		r.emitBlock("\n")
		r.emitBlock(r.fi(1, "{{gray}}Captured StdOut/StdErr Output >>{{/}}"))
		r.emitBlock(r.fi(1, "%s", report.CapturedStdOutErr))
		r.emitBlock(r.fi(1, "{{gray}}<< Captured StdOut/StdErr Output{{/}}"))
	}

	if showSeparateVisibilityAlwaysReportsSection {
		r.emitBlock("\n")
		r.emitBlock(r.fi(1, "{{gray}}Report Entries >>{{/}}"))
		for _, entry := range report.ReportEntries.WithVisibility(types.ReportEntryVisibilityAlways) {
			r.emitReportEntry(1, entry)
		}
		r.emitBlock(r.fi(1, "{{gray}}<< Report Entries{{/}}"))
	}

	if showTimeline {
		r.emitBlock("\n")
		r.emitBlock(r.fi(1, "{{gray}}Timeline >>{{/}}"))
		r.emitTimeline(1, report, timeline)
		r.emitBlock(r.fi(1, "{{gray}}<< Timeline{{/}}"))
	}

	// Emit Failure Message
	if !report.Failure.IsZero() && !v.Is(types.VerbosityLevelVeryVerbose) {
		r.emitBlock("\n")
		r.emitFailure(1, report.State, report.Failure, true)
		if len(report.AdditionalFailures) > 0 {
			r.emitBlock(r.fi(1, "\nThere were {{bold}}{{red}}additional failures{{/}} detected.  To view them in detail run {{bold}}ginkgo -vv{{/}}"))
		}
	}

	r.emitDelimiter(0)
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

func (r *DefaultReporter) emitTimeline(indent uint, report types.SpecReport, timeline types.Timeline) {
	isVeryVerbose := r.conf.Verbosity().Is(types.VerbosityLevelVeryVerbose)
	gw := report.CapturedGinkgoWriterOutput
	cursor := 0
	for _, entry := range timeline {
		tl := entry.GetTimelineLocation()
		if tl.Offset < len(gw) {
			r.emit(r.fi(indent, "%s", gw[cursor:tl.Offset]))
			cursor = tl.Offset
		} else if cursor < len(gw) {
			r.emit(r.fi(indent, "%s", gw[cursor:]))
			cursor = len(gw)
		}
		switch x := entry.(type) {
		case types.Failure:
			if isVeryVerbose {
				r.emitFailure(indent, report.State, x, false)
			} else {
				r.emitShortFailure(indent, report.State, x)
			}
		case types.AdditionalFailure:
			if isVeryVerbose {
				r.emitFailure(indent, x.State, x.Failure, true)
			} else {
				r.emitShortFailure(indent, x.State, x.Failure)
			}
		case types.ReportEntry:
			r.emitReportEntry(indent, x)
		case types.ProgressReport:
			r.emitProgressReport(indent, false, x)
		case types.SpecEvent:
			if isVeryVerbose || !x.IsOnlyVisibleAtVeryVerbose() || r.conf.ShowNodeEvents {
				r.emitSpecEvent(indent, x, isVeryVerbose)
			}
		}
	}
	if cursor < len(gw) {
		r.emit(r.fi(indent, "%s", gw[cursor:]))
	}
}

func (r *DefaultReporter) EmitFailure(state types.SpecState, failure types.Failure) {
	if r.conf.Verbosity().Is(types.VerbosityLevelVerbose) {
		r.emitShortFailure(1, state, failure)
	} else if r.conf.Verbosity().Is(types.VerbosityLevelVeryVerbose) {
		r.emitFailure(1, state, failure, true)
	}
}

func (r *DefaultReporter) emitShortFailure(indent uint, state types.SpecState, failure types.Failure) {
	r.emitBlock(r.fi(indent, r.highlightColorForState(state)+"[%s]{{/}} in [%s] - %s {{gray}}@ %s{{/}}",
		r.humanReadableState(state),
		failure.FailureNodeType,
		failure.Location,
		failure.TimelineLocation.Time.Format(types.GINKGO_TIME_FORMAT),
	))
}

func (r *DefaultReporter) emitFailure(indent uint, state types.SpecState, failure types.Failure, includeAdditionalFailure bool) {
	highlightColor := r.highlightColorForState(state)
	r.emitBlock(r.fi(indent, highlightColor+"[%s] %s{{/}}", r.humanReadableState(state), failure.Message))
	r.emitBlock(r.fi(indent, highlightColor+"In {{bold}}[%s]{{/}}"+highlightColor+" at: {{bold}}%s{{/}} {{gray}}@ %s{{/}}\n", failure.FailureNodeType, failure.Location, failure.TimelineLocation.Time.Format(types.GINKGO_TIME_FORMAT)))
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

	if failure.AdditionalFailure != nil && includeAdditionalFailure {
		r.emitBlock("\n")
		r.emitFailure(indent, failure.AdditionalFailure.State, failure.AdditionalFailure.Failure, true)
	}
}

func (r *DefaultReporter) EmitProgressReport(report types.ProgressReport) {
	r.emitDelimiter(1)

	if report.RunningInParallel {
		r.emit(r.fi(1, "{{coral}}Progress Report for Ginkgo Process #{{bold}}%d{{/}}\n", report.ParallelProcess))
	}
	shouldEmitGW := report.RunningInParallel || r.conf.Verbosity().LT(types.VerbosityLevelVerbose)
	r.emitProgressReport(1, shouldEmitGW, report)
	r.emitDelimiter(1)
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
		r.emit(r.fi(subjectIndent, "{{bold}}{{orange}}%s{{/}} (Spec Runtime: %s)\n", report.LeafNodeText, report.Time().Sub(report.SpecStartTime).Round(time.Millisecond)))
		r.emit(r.fi(indent+1, "{{gray}}%s{{/}}\n", report.LeafNodeLocation))
		indent += 1
	}
	if report.CurrentNodeType != types.NodeTypeInvalid {
		r.emit(r.fi(indent, "In {{bold}}{{orange}}[%s]{{/}}", report.CurrentNodeType))
		if report.CurrentNodeText != "" && !report.CurrentNodeType.Is(types.NodeTypeIt) {
			r.emit(r.f(" {{bold}}{{orange}}%s{{/}}", report.CurrentNodeText))
		}

		r.emit(r.f(" (Node Runtime: %s)\n", report.Time().Sub(report.CurrentNodeStartTime).Round(time.Millisecond)))
		r.emit(r.fi(indent+1, "{{gray}}%s{{/}}\n", report.CurrentNodeLocation))
		indent += 1
	}
	if report.CurrentStepText != "" {
		r.emit(r.fi(indent, "At {{bold}}{{orange}}[By Step] %s{{/}} (Step Runtime: %s)\n", report.CurrentStepText, report.Time().Sub(report.CurrentStepStartTime).Round(time.Millisecond)))
		r.emit(r.fi(indent+1, "{{gray}}%s{{/}}\n", report.CurrentStepLocation))
		indent += 1
	}

	if indent > 0 {
		indent -= 1
	}

	if emitGinkgoWriterOutput && report.CapturedGinkgoWriterOutput != "" {
		r.emit("\n")
		r.emitBlock(r.fi(indent, "{{gray}}Begin Captured GinkgoWriter Output >>{{/}}"))
		limit, lines := 10, strings.Split(report.CapturedGinkgoWriterOutput, "\n")
		if len(lines) <= limit {
			r.emitBlock(r.fi(indent+1, "%s", report.CapturedGinkgoWriterOutput))
		} else {
			r.emitBlock(r.fi(indent+1, "{{gray}}...{{/}}"))
			for _, line := range lines[len(lines)-limit-1:] {
				r.emitBlock(r.fi(indent+1, "%s", line))
			}
		}
		r.emitBlock(r.fi(indent, "{{gray}}<< End Captured GinkgoWriter Output{{/}}"))
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

func (r *DefaultReporter) EmitReportEntry(entry types.ReportEntry) {
	if r.conf.Verbosity().LT(types.VerbosityLevelVerbose) || entry.Visibility == types.ReportEntryVisibilityNever {
		return
	}
	r.emitReportEntry(1, entry)
}

func (r *DefaultReporter) emitReportEntry(indent uint, entry types.ReportEntry) {
	r.emitBlock(r.fi(indent, "{{bold}}"+entry.Name+"{{gray}} - %s @ %s{{/}}", entry.Location, entry.Time.Format(types.GINKGO_TIME_FORMAT)))
	if representation := entry.StringRepresentation(); representation != "" {
		r.emitBlock(r.fi(indent+1, representation))
	}
}

func (r *DefaultReporter) EmitSpecEvent(event types.SpecEvent) {
	v := r.conf.Verbosity()
	if v.Is(types.VerbosityLevelVeryVerbose) || (v.Is(types.VerbosityLevelVerbose) && (r.conf.ShowNodeEvents || !event.IsOnlyVisibleAtVeryVerbose())) {
		r.emitSpecEvent(1, event, r.conf.Verbosity().Is(types.VerbosityLevelVeryVerbose))
	}
}

func (r *DefaultReporter) emitSpecEvent(indent uint, event types.SpecEvent, includeLocation bool) {
	location := ""
	if includeLocation {
		location = fmt.Sprintf("- %s ", event.CodeLocation.String())
	}
	switch event.SpecEventType {
	case types.SpecEventInvalid:
		return
	case types.SpecEventByStart:
		r.emitBlock(r.fi(indent, "{{bold}}STEP:{{/}} %s {{gray}}%s@ %s{{/}}", event.Message, location, event.TimelineLocation.Time.Format(types.GINKGO_TIME_FORMAT)))
	case types.SpecEventByEnd:
		r.emitBlock(r.fi(indent, "{{bold}}END STEP:{{/}} %s {{gray}}%s@ %s (%s){{/}}", event.Message, location, event.TimelineLocation.Time.Format(types.GINKGO_TIME_FORMAT), event.Duration.Round(time.Millisecond)))
	case types.SpecEventNodeStart:
		r.emitBlock(r.fi(indent, "> Enter {{bold}}[%s]{{/}} %s {{gray}}%s@ %s{{/}}", event.NodeType.String(), event.Message, location, event.TimelineLocation.Time.Format(types.GINKGO_TIME_FORMAT)))
	case types.SpecEventNodeEnd:
		r.emitBlock(r.fi(indent, "< Exit {{bold}}[%s]{{/}} %s {{gray}}%s@ %s (%s){{/}}", event.NodeType.String(), event.Message, location, event.TimelineLocation.Time.Format(types.GINKGO_TIME_FORMAT), event.Duration.Round(time.Millisecond)))
	case types.SpecEventSpecRepeat:
		r.emitBlock(r.fi(indent, "\n{{bold}}Attempt #%d {{green}}Passed{{/}}{{bold}}.  Repeating %s{{/}} {{gray}}@ %s{{/}}\n\n", event.Attempt, r.retryDenoter, event.TimelineLocation.Time.Format(types.GINKGO_TIME_FORMAT)))
	case types.SpecEventSpecRetry:
		r.emitBlock(r.fi(indent, "\n{{bold}}Attempt #%d {{red}}Failed{{/}}{{bold}}.  Retrying %s{{/}} {{gray}}@ %s{{/}}\n\n", event.Attempt, r.retryDenoter, event.TimelineLocation.Time.Format(types.GINKGO_TIME_FORMAT)))
	}
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

func (r *DefaultReporter) emitDelimiter(indent uint) {
	if r.lastEmissionWasDelimiter {
		return
	}
	r.emitBlock(r.fi(indent, "{{gray}}%s{{/}}", strings.Repeat("-", 30)))
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

func (r *DefaultReporter) codeLocationBlock(report types.SpecReport, highlightColor string, veryVerbose bool, usePreciseFailureLocation bool) string {
	texts, locations, labels := []string{}, []types.CodeLocation{}, [][]string{}
	texts, locations, labels = append(texts, report.ContainerHierarchyTexts...), append(locations, report.ContainerHierarchyLocations...), append(labels, report.ContainerHierarchyLabels...)

	if report.LeafNodeType.Is(types.NodeTypesForSuiteLevelNodes) {
		texts = append(texts, r.f("[%s] %s", report.LeafNodeType, report.LeafNodeText))
	} else {
		texts = append(texts, r.f(report.LeafNodeText))
	}
	labels = append(labels, report.LeafNodeLabels)
	locations = append(locations, report.LeafNodeLocation)

	failureLocation := report.Failure.FailureNodeLocation
	if usePreciseFailureLocation {
		failureLocation = report.Failure.Location
	}

	highlightIndex := -1
	switch report.Failure.FailureNodeContext {
	case types.FailureNodeAtTopLevel:
		texts = append([]string{fmt.Sprintf("TOP-LEVEL [%s]", report.Failure.FailureNodeType)}, texts...)
		locations = append([]types.CodeLocation{failureLocation}, locations...)
		labels = append([][]string{{}}, labels...)
		highlightIndex = 0
	case types.FailureNodeInContainer:
		i := report.Failure.FailureNodeContainerIndex
		texts[i] = fmt.Sprintf("%s [%s]", texts[i], report.Failure.FailureNodeType)
		locations[i] = failureLocation
		highlightIndex = i
	case types.FailureNodeIsLeafNode:
		i := len(texts) - 1
		texts[i] = fmt.Sprintf("[%s] %s", report.LeafNodeType, report.LeafNodeText)
		locations[i] = failureLocation
		highlightIndex = i
	default:
		//there is no failure, so we highlight the leaf ndoe
		highlightIndex = len(texts) - 1
	}

	out := ""
	if veryVerbose {
		for i := range texts {
			if i == highlightIndex {
				out += r.fi(uint(i), highlightColor+"{{bold}}%s{{/}}", texts[i])
			} else {
				out += r.fi(uint(i), "%s", texts[i])
			}
			if len(labels[i]) > 0 {
				out += r.f(" {{coral}}[%s]{{/}}", strings.Join(labels[i], ", "))
			}
			out += "\n"
			out += r.fi(uint(i), "{{gray}}%s{{/}}\n", locations[i])
		}
	} else {
		for i := range texts {
			style := "{{/}}"
			if i%2 == 1 {
				style = "{{gray}}"
			}
			if i == highlightIndex {
				style = highlightColor + "{{bold}}"
			}
			out += r.f(style+"%s", texts[i])
			if i < len(texts)-1 {
				out += " "
			} else {
				out += r.f("{{/}}")
			}
		}
		flattenedLabels := report.Labels()
		if len(flattenedLabels) > 0 {
			out += r.f(" {{coral}}[%s]{{/}}", strings.Join(flattenedLabels, ", "))
		}
		out += "\n"
		if usePreciseFailureLocation {
			out += r.f("{{gray}}%s{{/}}", failureLocation)
		} else {
			leafLocation := locations[len(locations)-1]
			if (report.Failure.FailureNodeLocation != types.CodeLocation{}) && (report.Failure.FailureNodeLocation != leafLocation) {
				out += r.fi(1, highlightColor+"[%s]{{/}} {{gray}}%s{{/}}\n", report.Failure.FailureNodeType, report.Failure.FailureNodeLocation)
				out += r.fi(1, "{{gray}}[%s] %s{{/}}", report.LeafNodeType, leafLocation)
			} else {
				out += r.f("{{gray}}%s{{/}}", leafLocation)
			}
		}

	}
	return out
}
