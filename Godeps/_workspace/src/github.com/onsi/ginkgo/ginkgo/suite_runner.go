package main

import (
	"fmt"
	"runtime"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/ginkgo/interrupthandler"
	"github.com/onsi/ginkgo/ginkgo/testrunner"
	"github.com/onsi/ginkgo/ginkgo/testsuite"
)

type SuiteRunner struct {
	notifier         *Notifier
	interruptHandler *interrupthandler.InterruptHandler
}

type compiler struct {
	runner           *testrunner.TestRunner
	compilationError chan error
}

func (c *compiler) compile() {
	retries := 0

	err := c.runner.Compile()
	for err != nil && retries < 5 { //We retry because Go sometimes steps on itself when multiple compiles happen in parallel.  This is ugly, but should help resolve flakiness...
		err = c.runner.Compile()
		retries++
	}

	c.compilationError <- err
}

func NewSuiteRunner(notifier *Notifier, interruptHandler *interrupthandler.InterruptHandler) *SuiteRunner {
	return &SuiteRunner{
		notifier:         notifier,
		interruptHandler: interruptHandler,
	}
}

func (r *SuiteRunner) RunSuites(runners []*testrunner.TestRunner, numCompilers int, keepGoing bool, willCompile func(suite testsuite.TestSuite)) (testrunner.RunResult, int) {
	runResult := testrunner.PassingRunResult()

	compilers := make([]*compiler, len(runners))
	for i, runner := range runners {
		compilers[i] = &compiler{
			runner:           runner,
			compilationError: make(chan error, 1),
		}
	}

	compilerChannel := make(chan *compiler)
	if numCompilers == 0 {
		numCompilers = runtime.NumCPU()
	}
	for i := 0; i < numCompilers; i++ {
		go func() {
			for compiler := range compilerChannel {
				if willCompile != nil {
					willCompile(compiler.runner.Suite)
				}
				compiler.compile()
			}
		}()
	}
	go func() {
		for _, compiler := range compilers {
			compilerChannel <- compiler
		}
		close(compilerChannel)
	}()

	numSuitesThatRan := 0
	suitesThatFailed := []testsuite.TestSuite{}
	for i, runner := range runners {
		if r.interruptHandler.WasInterrupted() {
			break
		}

		compilationError := <-compilers[i].compilationError
		if compilationError != nil {
			fmt.Print(compilationError.Error())
		}
		numSuitesThatRan++
		suiteRunResult := testrunner.FailingRunResult()
		if compilationError == nil {
			suiteRunResult = compilers[i].runner.Run()
		}
		r.notifier.SendSuiteCompletionNotification(runner.Suite, suiteRunResult.Passed)
		runResult = runResult.Merge(suiteRunResult)
		if !suiteRunResult.Passed {
			suitesThatFailed = append(suitesThatFailed, runner.Suite)
			if !keepGoing {
				break
			}
		}
		if i < len(runners)-1 && !config.DefaultReporterConfig.Succinct {
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
			fmt.Printf("\t%s"+packageNameFormatter+"%s %s%s%s\n", redColor, suite.PackageName, defaultStyle, lightGrayColor, suite.Path, defaultStyle)
		}
	}
}
