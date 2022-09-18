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

	switch report.State {
	case types.SpecStatePassed:
		highlightColor, succinctLocationBlock = "{{green}}", v.LT(types.VerbosityLevelVerbose)
		emitGinkgoWriterOutput = (r.conf.AlwaysEmitGinkgoWriter || v.GTE(types.VerbosityLevelVerbose)) && hasGW
		if report.LeafNodeType.Is(types.NodeTypesForSuiteLevelNodes) {
			if v.GTE(types.VerbosityLevelVerbose) || hasStd || hasEmittableReports {
				header = fmt.Sprintf("%s PASSED", denoter)
			} else {
				return
			}
		} else {
			header, stream = denoter, true
			if report.NumAttempts > 1 {
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
		highlightColor = "{{yellow}}"
		includeRuntime, emitGinkgoWriterOutput = false, false
		if v.Is(types.VerbosityLevelSuccinct) {
			header, stream = "P", true
		} else {
			header, succinctLocationBlock = "P [PENDING]", v.LT(types.VerbosityLevelVeryVerbose)
		}
	case types.SpecStateSkipped:
		highlightColor = "{{cyan}}"
		if report.Failure.Message != "" || v.Is(types.VerbosityLevelVeryVerbose) {
			header = "S [SKIPPED]"
		} else {
			header, stream = "S", true
		}
	case types.SpecStateFailed:
		highlightColor, header = "{{red}}", fmt.Sprintf("%s [FAILED]", denoter)
	case types.SpecStatePanicked:
		highlightColor, header = "{{magenta}}", fmt.Sprintf("%s! [PANICKED]", denoter)
	case types.SpecStateInterrupted:
		highlightColor, header = "{{orange}}", fmt.Sprintf("%s! [INTERRUPTED]", denoter)
	case types.SpecStateAborted:
		highlightColor, header = "{{coral}}", fmt.Sprintf("%s! [ABORTED]", denoter)
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
		r.emitBlock(r.fi(1, "{{gray}}Begin Captured GinkgoWriter Output >>{{/}}"))
		r.emitBlock(r.fi(2, "%s", report.CapturedGinkgoWriterOutput))
		r.emitBlock(r.fi(1, "{{gray}}<< End Captured GinkgoWriter Output{{/}}"))
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
		r.emitBlock(r.fi(1, highlightColor+"%s{{/}}", report.Failure.Message))
		r.emitBlock(r.fi(1, highlightColor+"In {{bold}}[%s]{{/}}"+highlightColor+" at: {{bold}}%s{{/}}\n", report.Failure.FailureNodeType, report.Failure.Location))
		if report.Failure.ForwardedPanic != "" {
			r.emitBlock("\n")
			r.emitBlock(r.fi(1, highlightColor+"%s{{/}}", report.Failure.ForwardedPanic))
		}

		if r.conf.FullTrace || report.Failure.ForwardedPanic != "" {
			r.emitBlock("\n")
			r.emitBlock(r.fi(1, highlightColor+"Full Stack Trace{{/}}"))
			r.emitBlock(r.fi(2, "%s", report.Failure.Location.FullStackTrace))
		}
	}

	r.emitDelimiter()
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
		r.emit(r.f("{{yellow}}{{bold}}%d Pending{{/}} | ", specs.CountWithState(types.SpecStatePending)))
		r.emit(r.f("{{cyan}}{{bold}}%d Skipped{{/}}\n", specs.CountWithState(types.SpecStateSkipped)))
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
