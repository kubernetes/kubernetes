package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/ginkgo/interrupthandler"
	"github.com/onsi/ginkgo/ginkgo/testrunner"
	"github.com/onsi/ginkgo/ginkgo/testsuite"
	"github.com/onsi/ginkgo/ginkgo/watch"
	colorable "github.com/onsi/ginkgo/reporters/stenographer/support/go-colorable"
)

func BuildWatchCommand() *Command {
	commandFlags := NewWatchCommandFlags(flag.NewFlagSet("watch", flag.ExitOnError))
	interruptHandler := interrupthandler.NewInterruptHandler()
	notifier := NewNotifier(commandFlags)
	watcher := &SpecWatcher{
		commandFlags:     commandFlags,
		notifier:         notifier,
		interruptHandler: interruptHandler,
		suiteRunner:      NewSuiteRunner(notifier, interruptHandler),
	}

	return &Command{
		Name:         "watch",
		FlagSet:      commandFlags.FlagSet,
		UsageCommand: "ginkgo watch <FLAGS> <PACKAGES> -- <PASS-THROUGHS>",
		Usage: []string{
			"Watches the tests in the passed in <PACKAGES> and runs them when changes occur.",
			"Any arguments after -- will be passed to the test.",
		},
		Command:                   watcher.WatchSpecs,
		SuppressFlagDocumentation: true,
		FlagDocSubstitute: []string{
			"Accepts all the flags that the ginkgo command accepts except for --keepGoing and --untilItFails",
		},
	}
}

type SpecWatcher struct {
	commandFlags     *RunWatchAndBuildCommandFlags
	notifier         *Notifier
	interruptHandler *interrupthandler.InterruptHandler
	suiteRunner      *SuiteRunner
}

func (w *SpecWatcher) WatchSpecs(args []string, additionalArgs []string) {
	w.commandFlags.computeNodes()
	w.notifier.VerifyNotificationsAreAvailable()

	w.WatchSuites(args, additionalArgs)
}

func (w *SpecWatcher) runnersForSuites(suites []testsuite.TestSuite, additionalArgs []string) []*testrunner.TestRunner {
	runners := []*testrunner.TestRunner{}

	for _, suite := range suites {
		runners = append(runners, testrunner.New(suite, w.commandFlags.NumCPU, w.commandFlags.ParallelStream, w.commandFlags.GoOpts, additionalArgs))
	}

	return runners
}

func (w *SpecWatcher) WatchSuites(args []string, additionalArgs []string) {
	suites, _ := findSuites(args, w.commandFlags.Recurse, w.commandFlags.SkipPackage, false)

	if len(suites) == 0 {
		complainAndQuit("Found no test suites")
	}

	fmt.Printf("Identified %d test %s.  Locating dependencies to a depth of %d (this may take a while)...\n", len(suites), pluralizedWord("suite", "suites", len(suites)), w.commandFlags.Depth)
	deltaTracker := watch.NewDeltaTracker(w.commandFlags.Depth)
	delta, errors := deltaTracker.Delta(suites)

	fmt.Printf("Watching %d %s:\n", len(delta.NewSuites), pluralizedWord("suite", "suites", len(delta.NewSuites)))
	for _, suite := range delta.NewSuites {
		fmt.Println("  " + suite.Description())
	}

	for suite, err := range errors {
		fmt.Printf("Failed to watch %s: %s\n", suite.PackageName, err)
	}

	if len(suites) == 1 {
		runners := w.runnersForSuites(suites, additionalArgs)
		w.suiteRunner.RunSuites(runners, w.commandFlags.NumCompilers, true, nil)
		runners[0].CleanUp()
	}

	ticker := time.NewTicker(time.Second)

	for {
		select {
		case <-ticker.C:
			suites, _ := findSuites(args, w.commandFlags.Recurse, w.commandFlags.SkipPackage, false)
			delta, _ := deltaTracker.Delta(suites)
			coloredStream := colorable.NewColorableStdout()

			suitesToRun := []testsuite.TestSuite{}

			if len(delta.NewSuites) > 0 {
				fmt.Fprintf(coloredStream, greenColor+"Detected %d new %s:\n"+defaultStyle, len(delta.NewSuites), pluralizedWord("suite", "suites", len(delta.NewSuites)))
				for _, suite := range delta.NewSuites {
					suitesToRun = append(suitesToRun, suite.Suite)
					fmt.Fprintln(coloredStream, "  "+suite.Description())
				}
			}

			modifiedSuites := delta.ModifiedSuites()
			if len(modifiedSuites) > 0 {
				fmt.Fprintln(coloredStream, greenColor+"\nDetected changes in:"+defaultStyle)
				for _, pkg := range delta.ModifiedPackages {
					fmt.Fprintln(coloredStream, "  "+pkg)
				}
				fmt.Fprintf(coloredStream, greenColor+"Will run %d %s:\n"+defaultStyle, len(modifiedSuites), pluralizedWord("suite", "suites", len(modifiedSuites)))
				for _, suite := range modifiedSuites {
					suitesToRun = append(suitesToRun, suite.Suite)
					fmt.Fprintln(coloredStream, "  "+suite.Description())
				}
				fmt.Fprintln(coloredStream, "")
			}

			if len(suitesToRun) > 0 {
				w.UpdateSeed()
				w.ComputeSuccinctMode(len(suitesToRun))
				runners := w.runnersForSuites(suitesToRun, additionalArgs)
				result, _ := w.suiteRunner.RunSuites(runners, w.commandFlags.NumCompilers, true, func(suite testsuite.TestSuite) {
					deltaTracker.WillRun(suite)
				})
				for _, runner := range runners {
					runner.CleanUp()
				}
				if !w.interruptHandler.WasInterrupted() {
					color := redColor
					if result.Passed {
						color = greenColor
					}
					fmt.Fprintln(coloredStream, color+"\nDone.  Resuming watch..."+defaultStyle)
				}
			}

		case <-w.interruptHandler.C:
			return
		}
	}
}

func (w *SpecWatcher) ComputeSuccinctMode(numSuites int) {
	if config.DefaultReporterConfig.Verbose {
		config.DefaultReporterConfig.Succinct = false
		return
	}

	if w.commandFlags.wasSet("succinct") {
		return
	}

	if numSuites == 1 {
		config.DefaultReporterConfig.Succinct = false
	}

	if numSuites > 1 {
		config.DefaultReporterConfig.Succinct = true
	}
}

func (w *SpecWatcher) UpdateSeed() {
	if !w.commandFlags.wasSet("seed") {
		config.GinkgoConfig.RandomSeed = time.Now().Unix()
	}
}
