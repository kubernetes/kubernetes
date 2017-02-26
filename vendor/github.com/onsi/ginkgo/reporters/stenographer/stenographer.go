/*
The stenographer is used by Ginkgo's reporters to generate output.

Move along, nothing to see here.
*/

package stenographer

import (
	"fmt"
	"io"
	"runtime"
	"strings"

	"github.com/onsi/ginkgo/reporters/stenographer/support/go-colorable"
	"github.com/onsi/ginkgo/types"
)

const defaultStyle = "\x1b[0m"
const boldStyle = "\x1b[1m"
const redColor = "\x1b[91m"
const greenColor = "\x1b[32m"
const yellowColor = "\x1b[33m"
const cyanColor = "\x1b[36m"
const grayColor = "\x1b[90m"
const lightGrayColor = "\x1b[37m"

type cursorStateType int

const (
	cursorStateTop cursorStateType = iota
	cursorStateStreaming
	cursorStateMidBlock
	cursorStateEndBlock
)

type Stenographer interface {
	AnnounceSuite(description string, randomSeed int64, randomizingAll bool, succinct bool)
	AnnounceAggregatedParallelRun(nodes int, succinct bool)
	AnnounceParallelRun(node int, nodes int, specsToRun int, totalSpecs int, succinct bool)
	AnnounceNumberOfSpecs(specsToRun int, total int, succinct bool)
	AnnounceSpecRunCompletion(summary *types.SuiteSummary, succinct bool)

	AnnounceSpecWillRun(spec *types.SpecSummary)
	AnnounceBeforeSuiteFailure(summary *types.SetupSummary, succinct bool, fullTrace bool)
	AnnounceAfterSuiteFailure(summary *types.SetupSummary, succinct bool, fullTrace bool)

	AnnounceCapturedOutput(output string)

	AnnounceSuccesfulSpec(spec *types.SpecSummary)
	AnnounceSuccesfulSlowSpec(spec *types.SpecSummary, succinct bool)
	AnnounceSuccesfulMeasurement(spec *types.SpecSummary, succinct bool)

	AnnouncePendingSpec(spec *types.SpecSummary, noisy bool)
	AnnounceSkippedSpec(spec *types.SpecSummary, succinct bool, fullTrace bool)

	AnnounceSpecTimedOut(spec *types.SpecSummary, succinct bool, fullTrace bool)
	AnnounceSpecPanicked(spec *types.SpecSummary, succinct bool, fullTrace bool)
	AnnounceSpecFailed(spec *types.SpecSummary, succinct bool, fullTrace bool)

	SummarizeFailures(summaries []*types.SpecSummary)
}

func New(color bool, enableFlakes bool) Stenographer {
	denoter := "•"
	if runtime.GOOS == "windows" {
		denoter = "+"
	}
	return &consoleStenographer{
		color:        color,
		denoter:      denoter,
		cursorState:  cursorStateTop,
		enableFlakes: enableFlakes,
		w:            colorable.NewColorableStdout(),
	}
}

type consoleStenographer struct {
	color        bool
	denoter      string
	cursorState  cursorStateType
	enableFlakes bool
	w            io.Writer
}

var alternatingColors = []string{defaultStyle, grayColor}

func (s *consoleStenographer) AnnounceSuite(description string, randomSeed int64, randomizingAll bool, succinct bool) {
	if succinct {
		s.print(0, "[%d] %s ", randomSeed, s.colorize(boldStyle, description))
		return
	}
	s.printBanner(fmt.Sprintf("Running Suite: %s", description), "=")
	s.print(0, "Random Seed: %s", s.colorize(boldStyle, "%d", randomSeed))
	if randomizingAll {
		s.print(0, " - Will randomize all specs")
	}
	s.printNewLine()
}

func (s *consoleStenographer) AnnounceParallelRun(node int, nodes int, specsToRun int, totalSpecs int, succinct bool) {
	if succinct {
		s.print(0, "- node #%d ", node)
		return
	}
	s.println(0,
		"Parallel test node %s/%s. Assigned %s of %s specs.",
		s.colorize(boldStyle, "%d", node),
		s.colorize(boldStyle, "%d", nodes),
		s.colorize(boldStyle, "%d", specsToRun),
		s.colorize(boldStyle, "%d", totalSpecs),
	)
	s.printNewLine()
}

func (s *consoleStenographer) AnnounceAggregatedParallelRun(nodes int, succinct bool) {
	if succinct {
		s.print(0, "- %d nodes ", nodes)
		return
	}
	s.println(0,
		"Running in parallel across %s nodes",
		s.colorize(boldStyle, "%d", nodes),
	)
	s.printNewLine()
}

func (s *consoleStenographer) AnnounceNumberOfSpecs(specsToRun int, total int, succinct bool) {
	if succinct {
		s.print(0, "- %d/%d specs ", specsToRun, total)
		s.stream()
		return
	}
	s.println(0,
		"Will run %s of %s specs",
		s.colorize(boldStyle, "%d", specsToRun),
		s.colorize(boldStyle, "%d", total),
	)

	s.printNewLine()
}

func (s *consoleStenographer) AnnounceSpecRunCompletion(summary *types.SuiteSummary, succinct bool) {
	if succinct && summary.SuiteSucceeded {
		s.print(0, " %s %s ", s.colorize(greenColor, "SUCCESS!"), summary.RunTime)
		return
	}
	s.printNewLine()
	color := greenColor
	if !summary.SuiteSucceeded {
		color = redColor
	}
	s.println(0, s.colorize(boldStyle+color, "Ran %d of %d Specs in %.3f seconds", summary.NumberOfSpecsThatWillBeRun, summary.NumberOfTotalSpecs, summary.RunTime.Seconds()))

	status := ""
	if summary.SuiteSucceeded {
		status = s.colorize(boldStyle+greenColor, "SUCCESS!")
	} else {
		status = s.colorize(boldStyle+redColor, "FAIL!")
	}

	flakes := ""
	if s.enableFlakes {
		flakes = " | " + s.colorize(yellowColor+boldStyle, "%d Flaked", summary.NumberOfFlakedSpecs)
	}

	s.print(0,
		"%s -- %s | %s | %s | %s ",
		status,
		s.colorize(greenColor+boldStyle, "%d Passed", summary.NumberOfPassedSpecs),
		s.colorize(redColor+boldStyle, "%d Failed", summary.NumberOfFailedSpecs)+flakes,
		s.colorize(yellowColor+boldStyle, "%d Pending", summary.NumberOfPendingSpecs),
		s.colorize(cyanColor+boldStyle, "%d Skipped", summary.NumberOfSkippedSpecs),
	)
}

func (s *consoleStenographer) AnnounceSpecWillRun(spec *types.SpecSummary) {
	s.startBlock()
	for i, text := range spec.ComponentTexts[1 : len(spec.ComponentTexts)-1] {
		s.print(0, s.colorize(alternatingColors[i%2], text)+" ")
	}

	indentation := 0
	if len(spec.ComponentTexts) > 2 {
		indentation = 1
		s.printNewLine()
	}
	index := len(spec.ComponentTexts) - 1
	s.print(indentation, s.colorize(boldStyle, spec.ComponentTexts[index]))
	s.printNewLine()
	s.print(indentation, s.colorize(lightGrayColor, spec.ComponentCodeLocations[index].String()))
	s.printNewLine()
	s.midBlock()
}

func (s *consoleStenographer) AnnounceBeforeSuiteFailure(summary *types.SetupSummary, succinct bool, fullTrace bool) {
	s.announceSetupFailure("BeforeSuite", summary, succinct, fullTrace)
}

func (s *consoleStenographer) AnnounceAfterSuiteFailure(summary *types.SetupSummary, succinct bool, fullTrace bool) {
	s.announceSetupFailure("AfterSuite", summary, succinct, fullTrace)
}

func (s *consoleStenographer) announceSetupFailure(name string, summary *types.SetupSummary, succinct bool, fullTrace bool) {
	s.startBlock()
	var message string
	switch summary.State {
	case types.SpecStateFailed:
		message = "Failure"
	case types.SpecStatePanicked:
		message = "Panic"
	case types.SpecStateTimedOut:
		message = "Timeout"
	}

	s.println(0, s.colorize(redColor+boldStyle, "%s [%.3f seconds]", message, summary.RunTime.Seconds()))

	indentation := s.printCodeLocationBlock([]string{name}, []types.CodeLocation{summary.CodeLocation}, summary.ComponentType, 0, summary.State, true)

	s.printNewLine()
	s.printFailure(indentation, summary.State, summary.Failure, fullTrace)

	s.endBlock()
}

func (s *consoleStenographer) AnnounceCapturedOutput(output string) {
	if output == "" {
		return
	}

	s.startBlock()
	s.println(0, output)
	s.midBlock()
}

func (s *consoleStenographer) AnnounceSuccesfulSpec(spec *types.SpecSummary) {
	s.print(0, s.colorize(greenColor, s.denoter))
	s.stream()
}

func (s *consoleStenographer) AnnounceSuccesfulSlowSpec(spec *types.SpecSummary, succinct bool) {
	s.printBlockWithMessage(
		s.colorize(greenColor, "%s [SLOW TEST:%.3f seconds]", s.denoter, spec.RunTime.Seconds()),
		"",
		spec,
		succinct,
	)
}

func (s *consoleStenographer) AnnounceSuccesfulMeasurement(spec *types.SpecSummary, succinct bool) {
	s.printBlockWithMessage(
		s.colorize(greenColor, "%s [MEASUREMENT]", s.denoter),
		s.measurementReport(spec, succinct),
		spec,
		succinct,
	)
}

func (s *consoleStenographer) AnnouncePendingSpec(spec *types.SpecSummary, noisy bool) {
	if noisy {
		s.printBlockWithMessage(
			s.colorize(yellowColor, "P [PENDING]"),
			"",
			spec,
			false,
		)
	} else {
		s.print(0, s.colorize(yellowColor, "P"))
		s.stream()
	}
}

func (s *consoleStenographer) AnnounceSkippedSpec(spec *types.SpecSummary, succinct bool, fullTrace bool) {
	// Skips at runtime will have a non-empty spec.Failure. All others should be succinct.
	if succinct || spec.Failure == (types.SpecFailure{}) {
		s.print(0, s.colorize(cyanColor, "S"))
		s.stream()
	} else {
		s.startBlock()
		s.println(0, s.colorize(cyanColor+boldStyle, "S [SKIPPING]%s [%.3f seconds]", s.failureContext(spec.Failure.ComponentType), spec.RunTime.Seconds()))

		indentation := s.printCodeLocationBlock(spec.ComponentTexts, spec.ComponentCodeLocations, spec.Failure.ComponentType, spec.Failure.ComponentIndex, spec.State, succinct)

		s.printNewLine()
		s.printSkip(indentation, spec.Failure)
		s.endBlock()
	}
}

func (s *consoleStenographer) AnnounceSpecTimedOut(spec *types.SpecSummary, succinct bool, fullTrace bool) {
	s.printSpecFailure(fmt.Sprintf("%s... Timeout", s.denoter), spec, succinct, fullTrace)
}

func (s *consoleStenographer) AnnounceSpecPanicked(spec *types.SpecSummary, succinct bool, fullTrace bool) {
	s.printSpecFailure(fmt.Sprintf("%s! Panic", s.denoter), spec, succinct, fullTrace)
}

func (s *consoleStenographer) AnnounceSpecFailed(spec *types.SpecSummary, succinct bool, fullTrace bool) {
	s.printSpecFailure(fmt.Sprintf("%s Failure", s.denoter), spec, succinct, fullTrace)
}

func (s *consoleStenographer) SummarizeFailures(summaries []*types.SpecSummary) {
	failingSpecs := []*types.SpecSummary{}

	for _, summary := range summaries {
		if summary.HasFailureState() {
			failingSpecs = append(failingSpecs, summary)
		}
	}

	if len(failingSpecs) == 0 {
		return
	}

	s.printNewLine()
	s.printNewLine()
	plural := "s"
	if len(failingSpecs) == 1 {
		plural = ""
	}
	s.println(0, s.colorize(redColor+boldStyle, "Summarizing %d Failure%s:", len(failingSpecs), plural))
	for _, summary := range failingSpecs {
		s.printNewLine()
		if summary.HasFailureState() {
			if summary.TimedOut() {
				s.print(0, s.colorize(redColor+boldStyle, "[Timeout...] "))
			} else if summary.Panicked() {
				s.print(0, s.colorize(redColor+boldStyle, "[Panic!] "))
			} else if summary.Failed() {
				s.print(0, s.colorize(redColor+boldStyle, "[Fail] "))
			}
			s.printSpecContext(summary.ComponentTexts, summary.ComponentCodeLocations, summary.Failure.ComponentType, summary.Failure.ComponentIndex, summary.State, true)
			s.printNewLine()
			s.println(0, s.colorize(lightGrayColor, summary.Failure.Location.String()))
		}
	}
}

func (s *consoleStenographer) startBlock() {
	if s.cursorState == cursorStateStreaming {
		s.printNewLine()
		s.printDelimiter()
	} else if s.cursorState == cursorStateMidBlock {
		s.printNewLine()
	}
}

func (s *consoleStenographer) midBlock() {
	s.cursorState = cursorStateMidBlock
}

func (s *consoleStenographer) endBlock() {
	s.printDelimiter()
	s.cursorState = cursorStateEndBlock
}

func (s *consoleStenographer) stream() {
	s.cursorState = cursorStateStreaming
}

func (s *consoleStenographer) printBlockWithMessage(header string, message string, spec *types.SpecSummary, succinct bool) {
	s.startBlock()
	s.println(0, header)

	indentation := s.printCodeLocationBlock(spec.ComponentTexts, spec.ComponentCodeLocations, types.SpecComponentTypeInvalid, 0, spec.State, succinct)

	if message != "" {
		s.printNewLine()
		s.println(indentation, message)
	}

	s.endBlock()
}

func (s *consoleStenographer) printSpecFailure(message string, spec *types.SpecSummary, succinct bool, fullTrace bool) {
	s.startBlock()
	s.println(0, s.colorize(redColor+boldStyle, "%s%s [%.3f seconds]", message, s.failureContext(spec.Failure.ComponentType), spec.RunTime.Seconds()))

	indentation := s.printCodeLocationBlock(spec.ComponentTexts, spec.ComponentCodeLocations, spec.Failure.ComponentType, spec.Failure.ComponentIndex, spec.State, succinct)

	s.printNewLine()
	s.printFailure(indentation, spec.State, spec.Failure, fullTrace)
	s.endBlock()
}

func (s *consoleStenographer) failureContext(failedComponentType types.SpecComponentType) string {
	switch failedComponentType {
	case types.SpecComponentTypeBeforeSuite:
		return " in Suite Setup (BeforeSuite)"
	case types.SpecComponentTypeAfterSuite:
		return " in Suite Teardown (AfterSuite)"
	case types.SpecComponentTypeBeforeEach:
		return " in Spec Setup (BeforeEach)"
	case types.SpecComponentTypeJustBeforeEach:
		return " in Spec Setup (JustBeforeEach)"
	case types.SpecComponentTypeAfterEach:
		return " in Spec Teardown (AfterEach)"
	}

	return ""
}

func (s *consoleStenographer) printSkip(indentation int, spec types.SpecFailure) {
	s.println(indentation, s.colorize(cyanColor, spec.Message))
	s.printNewLine()
	s.println(indentation, spec.Location.String())
}

func (s *consoleStenographer) printFailure(indentation int, state types.SpecState, failure types.SpecFailure, fullTrace bool) {
	if state == types.SpecStatePanicked {
		s.println(indentation, s.colorize(redColor+boldStyle, failure.Message))
		s.println(indentation, s.colorize(redColor, failure.ForwardedPanic))
		s.println(indentation, failure.Location.String())
		s.printNewLine()
		s.println(indentation, s.colorize(redColor, "Full Stack Trace"))
		s.println(indentation, failure.Location.FullStackTrace)
	} else {
		s.println(indentation, s.colorize(redColor, failure.Message))
		s.printNewLine()
		s.println(indentation, failure.Location.String())
		if fullTrace {
			s.printNewLine()
			s.println(indentation, s.colorize(redColor, "Full Stack Trace"))
			s.println(indentation, failure.Location.FullStackTrace)
		}
	}
}

func (s *consoleStenographer) printSpecContext(componentTexts []string, componentCodeLocations []types.CodeLocation, failedComponentType types.SpecComponentType, failedComponentIndex int, state types.SpecState, succinct bool) int {
	startIndex := 1
	indentation := 0

	if len(componentTexts) == 1 {
		startIndex = 0
	}

	for i := startIndex; i < len(componentTexts); i++ {
		if (state.IsFailure() || state == types.SpecStateSkipped) && i == failedComponentIndex {
			color := redColor
			if state == types.SpecStateSkipped {
				color = cyanColor
			}
			blockType := ""
			switch failedComponentType {
			case types.SpecComponentTypeBeforeSuite:
				blockType = "BeforeSuite"
			case types.SpecComponentTypeAfterSuite:
				blockType = "AfterSuite"
			case types.SpecComponentTypeBeforeEach:
				blockType = "BeforeEach"
			case types.SpecComponentTypeJustBeforeEach:
				blockType = "JustBeforeEach"
			case types.SpecComponentTypeAfterEach:
				blockType = "AfterEach"
			case types.SpecComponentTypeIt:
				blockType = "It"
			case types.SpecComponentTypeMeasure:
				blockType = "Measurement"
			}
			if succinct {
				s.print(0, s.colorize(color+boldStyle, "[%s] %s ", blockType, componentTexts[i]))
			} else {
				s.println(indentation, s.colorize(color+boldStyle, "%s [%s]", componentTexts[i], blockType))
				s.println(indentation, s.colorize(grayColor, "%s", componentCodeLocations[i]))
			}
		} else {
			if succinct {
				s.print(0, s.colorize(alternatingColors[i%2], "%s ", componentTexts[i]))
			} else {
				s.println(indentation, componentTexts[i])
				s.println(indentation, s.colorize(grayColor, "%s", componentCodeLocations[i]))
			}
		}
		indentation++
	}

	return indentation
}

func (s *consoleStenographer) printCodeLocationBlock(componentTexts []string, componentCodeLocations []types.CodeLocation, failedComponentType types.SpecComponentType, failedComponentIndex int, state types.SpecState, succinct bool) int {
	indentation := s.printSpecContext(componentTexts, componentCodeLocations, failedComponentType, failedComponentIndex, state, succinct)

	if succinct {
		if len(componentTexts) > 0 {
			s.printNewLine()
			s.print(0, s.colorize(lightGrayColor, "%s", componentCodeLocations[len(componentCodeLocations)-1]))
		}
		s.printNewLine()
		indentation = 1
	} else {
		indentation--
	}

	return indentation
}

func (s *consoleStenographer) orderedMeasurementKeys(measurements map[string]*types.SpecMeasurement) []string {
	orderedKeys := make([]string, len(measurements))
	for key, measurement := range measurements {
		orderedKeys[measurement.Order] = key
	}
	return orderedKeys
}

func (s *consoleStenographer) measurementReport(spec *types.SpecSummary, succinct bool) string {
	if len(spec.Measurements) == 0 {
		return "Found no measurements"
	}

	message := []string{}
	orderedKeys := s.orderedMeasurementKeys(spec.Measurements)

	if succinct {
		message = append(message, fmt.Sprintf("%s samples:", s.colorize(boldStyle, "%d", spec.NumberOfSamples)))
		for _, key := range orderedKeys {
			measurement := spec.Measurements[key]
			message = append(message, fmt.Sprintf("  %s - %s: %s%s, %s: %s%s ± %s%s, %s: %s%s",
				s.colorize(boldStyle, "%s", measurement.Name),
				measurement.SmallestLabel,
				s.colorize(greenColor, measurement.PrecisionFmt(), measurement.Smallest),
				measurement.Units,
				measurement.AverageLabel,
				s.colorize(cyanColor, measurement.PrecisionFmt(), measurement.Average),
				measurement.Units,
				s.colorize(cyanColor, measurement.PrecisionFmt(), measurement.StdDeviation),
				measurement.Units,
				measurement.LargestLabel,
				s.colorize(redColor, measurement.PrecisionFmt(), measurement.Largest),
				measurement.Units,
			))
		}
	} else {
		message = append(message, fmt.Sprintf("Ran %s samples:", s.colorize(boldStyle, "%d", spec.NumberOfSamples)))
		for _, key := range orderedKeys {
			measurement := spec.Measurements[key]
			info := ""
			if measurement.Info != nil {
				message = append(message, fmt.Sprintf("%v", measurement.Info))
			}

			message = append(message, fmt.Sprintf("%s:\n%s  %s: %s%s\n  %s: %s%s\n  %s: %s%s ± %s%s",
				s.colorize(boldStyle, "%s", measurement.Name),
				info,
				measurement.SmallestLabel,
				s.colorize(greenColor, measurement.PrecisionFmt(), measurement.Smallest),
				measurement.Units,
				measurement.LargestLabel,
				s.colorize(redColor, measurement.PrecisionFmt(), measurement.Largest),
				measurement.Units,
				measurement.AverageLabel,
				s.colorize(cyanColor, measurement.PrecisionFmt(), measurement.Average),
				measurement.Units,
				s.colorize(cyanColor, measurement.PrecisionFmt(), measurement.StdDeviation),
				measurement.Units,
			))
		}
	}

	return strings.Join(message, "\n")
}
