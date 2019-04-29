package specrunner

import (
	"fmt"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/onsi/ginkgo/internal/spec_iterator"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/internal/leafnodes"
	"github.com/onsi/ginkgo/internal/spec"
	Writer "github.com/onsi/ginkgo/internal/writer"
	"github.com/onsi/ginkgo/reporters"
	"github.com/onsi/ginkgo/types"

	"time"
)

type SpecRunner struct {
	description     string
	beforeSuiteNode leafnodes.SuiteNode
	iterator        spec_iterator.SpecIterator
	afterSuiteNode  leafnodes.SuiteNode
	reporters       []reporters.Reporter
	startTime       time.Time
	suiteID         string
	runningSpec     *spec.Spec
	writer          Writer.WriterInterface
	config          config.GinkgoConfigType
	interrupted     bool
	processedSpecs  []*spec.Spec
	lock            *sync.Mutex
}

func New(description string, beforeSuiteNode leafnodes.SuiteNode, iterator spec_iterator.SpecIterator, afterSuiteNode leafnodes.SuiteNode, reporters []reporters.Reporter, writer Writer.WriterInterface, config config.GinkgoConfigType) *SpecRunner {
	return &SpecRunner{
		description:     description,
		beforeSuiteNode: beforeSuiteNode,
		iterator:        iterator,
		afterSuiteNode:  afterSuiteNode,
		reporters:       reporters,
		writer:          writer,
		config:          config,
		suiteID:         randomID(),
		lock:            &sync.Mutex{},
	}
}

func (runner *SpecRunner) Run() bool {
	if runner.config.DryRun {
		runner.performDryRun()
		return true
	}

	runner.reportSuiteWillBegin()
	signalRegistered := make(chan struct{})
	go runner.registerForInterrupts(signalRegistered)
	<-signalRegistered

	suitePassed := runner.runBeforeSuite()

	if suitePassed {
		suitePassed = runner.runSpecs()
	}

	runner.blockForeverIfInterrupted()

	suitePassed = runner.runAfterSuite() && suitePassed

	runner.reportSuiteDidEnd(suitePassed)

	return suitePassed
}

func (runner *SpecRunner) performDryRun() {
	runner.reportSuiteWillBegin()

	if runner.beforeSuiteNode != nil {
		summary := runner.beforeSuiteNode.Summary()
		summary.State = types.SpecStatePassed
		runner.reportBeforeSuite(summary)
	}

	for {
		spec, err := runner.iterator.Next()
		if err == spec_iterator.ErrClosed {
			break
		}
		if err != nil {
			fmt.Println("failed to iterate over tests:\n" + err.Error())
			break
		}

		runner.processedSpecs = append(runner.processedSpecs, spec)

		summary := spec.Summary(runner.suiteID)
		runner.reportSpecWillRun(summary)
		if summary.State == types.SpecStateInvalid {
			summary.State = types.SpecStatePassed
		}
		runner.reportSpecDidComplete(summary, false)
	}

	if runner.afterSuiteNode != nil {
		summary := runner.afterSuiteNode.Summary()
		summary.State = types.SpecStatePassed
		runner.reportAfterSuite(summary)
	}

	runner.reportSuiteDidEnd(true)
}

func (runner *SpecRunner) runBeforeSuite() bool {
	if runner.beforeSuiteNode == nil || runner.wasInterrupted() {
		return true
	}

	runner.writer.Truncate()
	conf := runner.config
	passed := runner.beforeSuiteNode.Run(conf.ParallelNode, conf.ParallelTotal, conf.SyncHost)
	if !passed {
		runner.writer.DumpOut()
	}
	runner.reportBeforeSuite(runner.beforeSuiteNode.Summary())
	return passed
}

func (runner *SpecRunner) runAfterSuite() bool {
	if runner.afterSuiteNode == nil {
		return true
	}

	runner.writer.Truncate()
	conf := runner.config
	passed := runner.afterSuiteNode.Run(conf.ParallelNode, conf.ParallelTotal, conf.SyncHost)
	if !passed {
		runner.writer.DumpOut()
	}
	runner.reportAfterSuite(runner.afterSuiteNode.Summary())
	return passed
}

func (runner *SpecRunner) runSpecs() bool {
	suiteFailed := false
	skipRemainingSpecs := false
	for {
		spec, err := runner.iterator.Next()
		if err == spec_iterator.ErrClosed {
			break
		}
		if err != nil {
			fmt.Println("failed to iterate over tests:\n" + err.Error())
			suiteFailed = true
			break
		}

		runner.processedSpecs = append(runner.processedSpecs, spec)

		if runner.wasInterrupted() {
			break
		}
		if skipRemainingSpecs {
			spec.Skip()
		}

		if !spec.Skipped() && !spec.Pending() {
			if passed := runner.runSpec(spec); !passed {
				suiteFailed = true
			}
		} else if spec.Pending() && runner.config.FailOnPending {
			runner.reportSpecWillRun(spec.Summary(runner.suiteID))
			suiteFailed = true
			runner.reportSpecDidComplete(spec.Summary(runner.suiteID), spec.Failed())
		} else {
			runner.reportSpecWillRun(spec.Summary(runner.suiteID))
			runner.reportSpecDidComplete(spec.Summary(runner.suiteID), spec.Failed())
		}

		if spec.Failed() && runner.config.FailFast {
			skipRemainingSpecs = true
		}
	}

	return !suiteFailed
}

func (runner *SpecRunner) runSpec(spec *spec.Spec) (passed bool) {
	maxAttempts := 1
	if runner.config.FlakeAttempts > 0 {
		// uninitialized configs count as 1
		maxAttempts = runner.config.FlakeAttempts
	}

	for i := 0; i < maxAttempts; i++ {
		runner.reportSpecWillRun(spec.Summary(runner.suiteID))
		runner.runningSpec = spec
		spec.Run(runner.writer)
		runner.runningSpec = nil
		runner.reportSpecDidComplete(spec.Summary(runner.suiteID), spec.Failed())
		if !spec.Failed() {
			return true
		}
	}
	return false
}

func (runner *SpecRunner) CurrentSpecSummary() (*types.SpecSummary, bool) {
	if runner.runningSpec == nil {
		return nil, false
	}

	return runner.runningSpec.Summary(runner.suiteID), true
}

func (runner *SpecRunner) registerForInterrupts(signalRegistered chan struct{}) {
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	close(signalRegistered)

	<-c
	signal.Stop(c)
	runner.markInterrupted()
	go runner.registerForHardInterrupts()
	runner.writer.DumpOutWithHeader(`
Received interrupt.  Emitting contents of GinkgoWriter...
---------------------------------------------------------
`)
	if runner.afterSuiteNode != nil {
		fmt.Fprint(os.Stderr, `
---------------------------------------------------------
Received interrupt.  Running AfterSuite...
^C again to terminate immediately
`)
		runner.runAfterSuite()
	}
	runner.reportSuiteDidEnd(false)
	os.Exit(1)
}

func (runner *SpecRunner) registerForHardInterrupts() {
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	<-c
	fmt.Fprintln(os.Stderr, "\nReceived second interrupt.  Shutting down.")
	os.Exit(1)
}

func (runner *SpecRunner) blockForeverIfInterrupted() {
	runner.lock.Lock()
	interrupted := runner.interrupted
	runner.lock.Unlock()

	if interrupted {
		select {}
	}
}

func (runner *SpecRunner) markInterrupted() {
	runner.lock.Lock()
	defer runner.lock.Unlock()
	runner.interrupted = true
}

func (runner *SpecRunner) wasInterrupted() bool {
	runner.lock.Lock()
	defer runner.lock.Unlock()
	return runner.interrupted
}

func (runner *SpecRunner) reportSuiteWillBegin() {
	runner.startTime = time.Now()
	summary := runner.suiteWillBeginSummary()
	for _, reporter := range runner.reporters {
		reporter.SpecSuiteWillBegin(runner.config, summary)
	}
}

func (runner *SpecRunner) reportBeforeSuite(summary *types.SetupSummary) {
	for _, reporter := range runner.reporters {
		reporter.BeforeSuiteDidRun(summary)
	}
}

func (runner *SpecRunner) reportAfterSuite(summary *types.SetupSummary) {
	for _, reporter := range runner.reporters {
		reporter.AfterSuiteDidRun(summary)
	}
}

func (runner *SpecRunner) reportSpecWillRun(summary *types.SpecSummary) {
	runner.writer.Truncate()

	for _, reporter := range runner.reporters {
		reporter.SpecWillRun(summary)
	}
}

func (runner *SpecRunner) reportSpecDidComplete(summary *types.SpecSummary, failed bool) {
	if failed && len(summary.CapturedOutput) == 0 {
		summary.CapturedOutput = string(runner.writer.Bytes())
	}
	for i := len(runner.reporters) - 1; i >= 1; i-- {
		runner.reporters[i].SpecDidComplete(summary)
	}

	if failed {
		runner.writer.DumpOut()
	}

	runner.reporters[0].SpecDidComplete(summary)
}

func (runner *SpecRunner) reportSuiteDidEnd(success bool) {
	summary := runner.suiteDidEndSummary(success)
	summary.RunTime = time.Since(runner.startTime)
	for _, reporter := range runner.reporters {
		reporter.SpecSuiteDidEnd(summary)
	}
}

func (runner *SpecRunner) countSpecsThatRanSatisfying(filter func(ex *spec.Spec) bool) (count int) {
	count = 0

	for _, spec := range runner.processedSpecs {
		if filter(spec) {
			count++
		}
	}

	return count
}

func (runner *SpecRunner) suiteDidEndSummary(success bool) *types.SuiteSummary {
	numberOfSpecsThatWillBeRun := runner.countSpecsThatRanSatisfying(func(ex *spec.Spec) bool {
		return !ex.Skipped() && !ex.Pending()
	})

	numberOfPendingSpecs := runner.countSpecsThatRanSatisfying(func(ex *spec.Spec) bool {
		return ex.Pending()
	})

	numberOfSkippedSpecs := runner.countSpecsThatRanSatisfying(func(ex *spec.Spec) bool {
		return ex.Skipped()
	})

	numberOfPassedSpecs := runner.countSpecsThatRanSatisfying(func(ex *spec.Spec) bool {
		return ex.Passed()
	})

	numberOfFlakedSpecs := runner.countSpecsThatRanSatisfying(func(ex *spec.Spec) bool {
		return ex.Flaked()
	})

	numberOfFailedSpecs := runner.countSpecsThatRanSatisfying(func(ex *spec.Spec) bool {
		return ex.Failed()
	})

	if runner.beforeSuiteNode != nil && !runner.beforeSuiteNode.Passed() && !runner.config.DryRun {
		var known bool
		numberOfSpecsThatWillBeRun, known = runner.iterator.NumberOfSpecsThatWillBeRunIfKnown()
		if !known {
			numberOfSpecsThatWillBeRun = runner.iterator.NumberOfSpecsPriorToIteration()
		}
		numberOfFailedSpecs = numberOfSpecsThatWillBeRun
	}

	return &types.SuiteSummary{
		SuiteDescription: runner.description,
		SuiteSucceeded:   success,
		SuiteID:          runner.suiteID,

		NumberOfSpecsBeforeParallelization: runner.iterator.NumberOfSpecsPriorToIteration(),
		NumberOfTotalSpecs:                 len(runner.processedSpecs),
		NumberOfSpecsThatWillBeRun:         numberOfSpecsThatWillBeRun,
		NumberOfPendingSpecs:               numberOfPendingSpecs,
		NumberOfSkippedSpecs:               numberOfSkippedSpecs,
		NumberOfPassedSpecs:                numberOfPassedSpecs,
		NumberOfFailedSpecs:                numberOfFailedSpecs,
		NumberOfFlakedSpecs:                numberOfFlakedSpecs,
	}
}

func (runner *SpecRunner) suiteWillBeginSummary() *types.SuiteSummary {
	numTotal, known := runner.iterator.NumberOfSpecsToProcessIfKnown()
	if !known {
		numTotal = -1
	}

	numToRun, known := runner.iterator.NumberOfSpecsThatWillBeRunIfKnown()
	if !known {
		numToRun = -1
	}

	return &types.SuiteSummary{
		SuiteDescription: runner.description,
		SuiteID:          runner.suiteID,

		NumberOfSpecsBeforeParallelization: runner.iterator.NumberOfSpecsPriorToIteration(),
		NumberOfTotalSpecs:                 numTotal,
		NumberOfSpecsThatWillBeRun:         numToRun,
		NumberOfPendingSpecs:               -1,
		NumberOfSkippedSpecs:               -1,
		NumberOfPassedSpecs:                -1,
		NumberOfFailedSpecs:                -1,
		NumberOfFlakedSpecs:                -1,
	}
}
