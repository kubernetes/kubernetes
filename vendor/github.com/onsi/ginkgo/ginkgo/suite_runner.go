package main

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/ginkgo/interrupthandler"
	"github.com/onsi/ginkgo/ginkgo/testrunner"
	"github.com/onsi/ginkgo/ginkgo/testsuite"
	colorable "github.com/onsi/ginkgo/reporters/stenographer/support/go-colorable"
)

type compilationInput struct {
	runner *testrunner.TestRunner
	result chan compilationOutput
}

type compilationOutput struct {
	runner *testrunner.TestRunner
	err    error
}

type SuiteRunner struct {
	notifier         *Notifier
	interruptHandler *interrupthandler.InterruptHandler
}

func NewSuiteRunner(notifier *Notifier, interruptHandler *interrupthandler.InterruptHandler) *SuiteRunner {
	return &SuiteRunner{
		notifier:         notifier,
		interruptHandler: interruptHandler,
	}
}

func (r *SuiteRunner) compileInParallel(runners []*testrunner.TestRunner, numCompilers int, willCompile func(suite testsuite.TestSuite)) chan compilationOutput {
	//we return this to the consumer, it will return each runner in order as it compiles
	compilationOutputs := make(chan compilationOutput, len(runners))

	//an array of channels - the nth runner's compilation output is sent to the nth channel in this array
	//we read from these channels in order to ensure we run the suites in order
	orderedCompilationOutputs := []chan compilationOutput{}
	for range runners {
		orderedCompilationOutputs = append(orderedCompilationOutputs, make(chan compilationOutput, 1))
	}

	//we're going to spin up numCompilers compilers - they're going to run concurrently and will consume this channel
	//we prefill the channel then close it, this ensures we compile things in the correct order
	workPool := make(chan compilationInput, len(runners))
	for i, runner := range runners {
		workPool <- compilationInput{runner, orderedCompilationOutputs[i]}
	}
	close(workPool)

	//pick a reasonable numCompilers
	if numCompilers == 0 {
		numCompilers = runtime.NumCPU()
	}

	//a WaitGroup to help us wait for all compilers to shut down
	wg := &sync.WaitGroup{}
	wg.Add(numCompilers)

	//spin up the concurrent compilers
	for i := 0; i < numCompilers; i++ {
		go func() {
			defer wg.Done()
			for input := range workPool {
				if r.interruptHandler.WasInterrupted() {
					return
				}

				if willCompile != nil {
					willCompile(input.runner.Suite)
				}

				//We retry because Go sometimes steps on itself when multiple compiles happen in parallel.  This is ugly, but should help resolve flakiness...
				var err error
				retries := 0
				for retries <= 5 {
					if r.interruptHandler.WasInterrupted() {
						return
					}
					if err = input.runner.Compile(); err == nil {
						break
					}
					retries++
				}

				input.result <- compilationOutput{input.runner, err}
			}
		}()
	}

	//read from the compilation output channels *in order* and send them to the caller
	//close the compilationOutputs channel to tell the caller we're done
	go func() {
		defer close(compilationOutputs)
		for _, orderedCompilationOutput := range orderedCompilationOutputs {
			select {
			case compilationOutput := <-orderedCompilationOutput:
				compilationOutputs <- compilationOutput
			case <-r.interruptHandler.C:
				//interrupt detected, wait for the compilers to shut down then bail
				//this ensure we clean up after ourselves as we don't leave any compilation processes running
				wg.Wait()
				return
			}
		}
	}()

	return compilationOutputs
}

func (r *SuiteRunner) RunSuites(runners []*testrunner.TestRunner, numCompilers int, keepGoing bool, willCompile func(suite testsuite.TestSuite)) (testrunner.RunResult, int) {
	runResult := testrunner.PassingRunResult()

	compilationOutputs := r.compileInParallel(runners, numCompilers, willCompile)

	numSuitesThatRan := 0
	suitesThatFailed := []testsuite.TestSuite{}
	for compilationOutput := range compilationOutputs {
		if compilationOutput.err != nil {
			fmt.Print(compilationOutput.err.Error())
		}
		numSuitesThatRan++
		suiteRunResult := testrunner.FailingRunResult()
		if compilationOutput.err == nil {
			suiteRunResult = compilationOutput.runner.Run()
		}
		r.notifier.SendSuiteCompletionNotification(compilationOutput.runner.Suite, suiteRunResult.Passed)
		r.notifier.RunCommand(compilationOutput.runner.Suite, suiteRunResult.Passed)
		runResult = runResult.Merge(suiteRunResult)
		if !suiteRunResult.Passed {
			suitesThatFailed = append(suitesThatFailed, compilationOutput.runner.Suite)
			if !keepGoing {
				break
			}
		}
		if numSuitesThatRan < len(runners) && !config.DefaultReporterConfig.Succinct {
			fmt.Println("")
		}
	}

	if keepGoing && !runResult.Passed {
		r.listFailedSuites(suitesThatFailed)
	}

	return runResult, numSuitesThatRan
}

func (r *SuiteRunner) listFailedSuites(suitesThatFailed []testsuite.TestSuite) {
	fmt.Println("")
	fmt.Println("There were failures detected in the following suites:")

	maxPackageNameLength := 0
	for _, suite := range suitesThatFailed {
		if len(suite.PackageName) > maxPackageNameLength {
			maxPackageNameLength = len(suite.PackageName)
		}
	}

	packageNameFormatter := fmt.Sprintf("%%%ds", maxPackageNameLength)

	for _, suite := range suitesThatFailed {
		if config.DefaultReporterConfig.NoColor {
			fmt.Printf("\t"+packageNameFormatter+" %s\n", suite.PackageName, suite.Path)
		} else {
			fmt.Fprintf(colorable.NewColorableStdout(), "\t%s"+packageNameFormatter+"%s %s%s%s\n", redColor, suite.PackageName, defaultStyle, lightGrayColor, suite.Path, defaultStyle)
		}
	}
}
