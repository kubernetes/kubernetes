package internal

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2/types"
)

var _SOURCE_CACHE = map[string][]string{}

type ProgressSignalRegistrar func(func()) context.CancelFunc

func RegisterForProgressSignal(handler func()) context.CancelFunc {
	signalChannel := make(chan os.Signal, 1)
	if len(PROGRESS_SIGNALS) > 0 {
		signal.Notify(signalChannel, PROGRESS_SIGNALS...)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		for {
			select {
			case <-signalChannel:
				handler()
			case <-ctx.Done():
				signal.Stop(signalChannel)
				return
			}
		}
	}()

	return cancel
}

type ProgressStepCursor struct {
	Text         string
	CodeLocation types.CodeLocation
	StartTime    time.Time
}

func NewProgressReport(isRunningInParallel bool, report types.SpecReport, currentNode Node, currentNodeStartTime time.Time, currentStep types.SpecEvent, gwOutput string, timelineLocation types.TimelineLocation, additionalReports []string, sourceRoots []string, includeAll bool) (types.ProgressReport, error) {
	pr := types.ProgressReport{
		ParallelProcess:         report.ParallelProcess,
		RunningInParallel:       isRunningInParallel,
		ContainerHierarchyTexts: report.ContainerHierarchyTexts,
		LeafNodeText:            report.LeafNodeText,
		LeafNodeLocation:        report.LeafNodeLocation,
		SpecStartTime:           report.StartTime,

		CurrentNodeType:      currentNode.NodeType,
		CurrentNodeText:      currentNode.Text,
		CurrentNodeLocation:  currentNode.CodeLocation,
		CurrentNodeStartTime: currentNodeStartTime,

		CurrentStepText:      currentStep.Message,
		CurrentStepLocation:  currentStep.CodeLocation,
		CurrentStepStartTime: currentStep.TimelineLocation.Time,

		AdditionalReports: additionalReports,

		CapturedGinkgoWriterOutput: gwOutput,
		TimelineLocation:           timelineLocation,
	}

	goroutines, err := extractRunningGoroutines()
	if err != nil {
		return pr, err
	}
	pr.Goroutines = goroutines

	// now we want to try to find goroutines of interest.  these will be goroutines that have any function calls with code in packagesOfInterest:
	packagesOfInterest := map[string]bool{}
	packageFromFilename := func(filename string) string {
		return filepath.Dir(filename)
	}
	addPackageFor := func(filename string) {
		if filename != "" {
			packagesOfInterest[packageFromFilename(filename)] = true
		}
	}
	isPackageOfInterest := func(filename string) bool {
		stackPackage := packageFromFilename(filename)
		for packageOfInterest := range packagesOfInterest {
			if strings.HasPrefix(stackPackage, packageOfInterest) {
				return true
			}
		}
		return false
	}
	for _, location := range report.ContainerHierarchyLocations {
		addPackageFor(location.FileName)
	}
	addPackageFor(report.LeafNodeLocation.FileName)
	addPackageFor(currentNode.CodeLocation.FileName)
	addPackageFor(currentStep.CodeLocation.FileName)

	//First, we find the SpecGoroutine - this will be the goroutine that includes `runNode`
	specGoRoutineIdx := -1
	runNodeFunctionCallIdx := -1
OUTER:
	for goroutineIdx, goroutine := range pr.Goroutines {
		for functionCallIdx, functionCall := range goroutine.Stack {
			if strings.Contains(functionCall.Function, "ginkgo/v2/internal.(*Suite).runNode.func") {
				specGoRoutineIdx = goroutineIdx
				runNodeFunctionCallIdx = functionCallIdx
				break OUTER
			}
		}
	}

	//Now, we find the first non-Ginkgo function call
	if specGoRoutineIdx > -1 {
		for runNodeFunctionCallIdx >= 0 {
			fn := goroutines[specGoRoutineIdx].Stack[runNodeFunctionCallIdx].Function
			file := goroutines[specGoRoutineIdx].Stack[runNodeFunctionCallIdx].Filename
			// these are all things that could potentially happen from within ginkgo
			if strings.Contains(fn, "ginkgo/v2/internal") || strings.Contains(fn, "reflect.Value") || strings.Contains(file, "ginkgo/table_dsl") || strings.Contains(file, "ginkgo/core_dsl") {
				runNodeFunctionCallIdx--
				continue
			}
			if strings.Contains(goroutines[specGoRoutineIdx].Stack[runNodeFunctionCallIdx].Function, "ginkgo/table_dsl") {

			}
			//found it!  lets add its package of interest
			addPackageFor(goroutines[specGoRoutineIdx].Stack[runNodeFunctionCallIdx].Filename)
			break
		}
	}

	ginkgoEntryPointIdx := -1
OUTER_GINKGO_ENTRY_POINT:
	for goroutineIdx, goroutine := range pr.Goroutines {
		for _, functionCall := range goroutine.Stack {
			if strings.Contains(functionCall.Function, "ginkgo/v2.RunSpecs") {
				ginkgoEntryPointIdx = goroutineIdx
				break OUTER_GINKGO_ENTRY_POINT
			}
		}
	}

	// Now we go through all goroutines and highlight any lines with packages in `packagesOfInterest`
	// Any goroutines with highlighted lines end up in the HighlightGoRoutines
	for goroutineIdx, goroutine := range pr.Goroutines {
		if goroutineIdx == ginkgoEntryPointIdx {
			continue
		}
		if goroutineIdx == specGoRoutineIdx {
			pr.Goroutines[goroutineIdx].IsSpecGoroutine = true
		}
		for functionCallIdx, functionCall := range goroutine.Stack {
			if isPackageOfInterest(functionCall.Filename) {
				goroutine.Stack[functionCallIdx].Highlight = true
				goroutine.Stack[functionCallIdx].Source, goroutine.Stack[functionCallIdx].SourceHighlight = fetchSource(functionCall.Filename, functionCall.Line, 2, sourceRoots)
			}
		}
	}

	if !includeAll {
		goroutines := []types.Goroutine{pr.SpecGoroutine()}
		goroutines = append(goroutines, pr.HighlightedGoroutines()...)
		pr.Goroutines = goroutines
	}

	return pr, nil
}

func extractRunningGoroutines() ([]types.Goroutine, error) {
	var stack []byte
	for size := 64 * 1024; ; size *= 2 {
		stack = make([]byte, size)
		if n := runtime.Stack(stack, true); n < size {
			stack = stack[:n]
			break
		}
	}
	r := bufio.NewReader(bytes.NewReader(stack))
	out := []types.Goroutine{}
	idx := -1
	for {
		line, err := r.ReadString('\n')
		if err == io.EOF {
			break
		}

		line = strings.TrimSuffix(line, "\n")

		//skip blank lines
		if line == "" {
			continue
		}

		//parse headers for new goroutine frames
		if strings.HasPrefix(line, "goroutine") {
			out = append(out, types.Goroutine{})
			idx = len(out) - 1

			line = strings.TrimPrefix(line, "goroutine ")
			line = strings.TrimSuffix(line, ":")
			fields := strings.SplitN(line, " ", 2)
			if len(fields) != 2 {
				return nil, types.GinkgoErrors.FailedToParseStackTrace(fmt.Sprintf("Invalid goroutine frame header: %s", line))
			}
			out[idx].ID, err = strconv.ParseUint(fields[0], 10, 64)
			if err != nil {
				return nil, types.GinkgoErrors.FailedToParseStackTrace(fmt.Sprintf("Invalid goroutine ID: %s", fields[1]))
			}

			out[idx].State = strings.TrimSuffix(strings.TrimPrefix(fields[1], "["), "]")
			continue
		}

		//if we are here we must be at a function call entry in the stack
		functionCall := types.FunctionCall{
			Function: strings.TrimPrefix(line, "created by "), // no need to track 'created by'
		}

		line, err = r.ReadString('\n')
		line = strings.TrimSuffix(line, "\n")
		if err == io.EOF {
			return nil, types.GinkgoErrors.FailedToParseStackTrace(fmt.Sprintf("Invalid function call: %s -- missing file name and line number", functionCall.Function))
		}
		line = strings.TrimLeft(line, " \t")
		delimiterIdx := strings.LastIndex(line, ":")
		if delimiterIdx == -1 {
			return nil, types.GinkgoErrors.FailedToParseStackTrace(fmt.Sprintf("Invalid filename and line number: %s", line))
		}
		functionCall.Filename = line[:delimiterIdx]
		line = strings.Split(line[delimiterIdx+1:], " ")[0]
		lineNumber, err := strconv.ParseInt(line, 10, 64)
		functionCall.Line = int(lineNumber)
		if err != nil {
			return nil, types.GinkgoErrors.FailedToParseStackTrace(fmt.Sprintf("Invalid function call line number: %s\n%s", line, err.Error()))
		}
		out[idx].Stack = append(out[idx].Stack, functionCall)
	}

	return out, nil
}

func fetchSource(filename string, lineNumber int, span int, configuredSourceRoots []string) ([]string, int) {
	if filename == "" {
		return []string{}, 0
	}

	var lines []string
	var ok bool
	if lines, ok = _SOURCE_CACHE[filename]; !ok {
		sourceRoots := []string{""}
		sourceRoots = append(sourceRoots, configuredSourceRoots...)
		var data []byte
		var err error
		var found bool
		for _, root := range sourceRoots {
			data, err = os.ReadFile(filepath.Join(root, filename))
			if err == nil {
				found = true
				break
			}
		}
		if !found {
			return []string{}, 0
		}
		lines = strings.Split(string(data), "\n")
		_SOURCE_CACHE[filename] = lines
	}

	startIndex := lineNumber - span - 1
	endIndex := startIndex + span + span + 1
	if startIndex < 0 {
		startIndex = 0
	}
	if endIndex > len(lines) {
		endIndex = len(lines)
	}
	highlightIndex := lineNumber - 1 - startIndex
	return lines[startIndex:endIndex], highlightIndex
}
