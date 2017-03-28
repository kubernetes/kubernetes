package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/ginkgo/interrupthandler"
	"github.com/onsi/ginkgo/ginkgo/testrunner"
	"github.com/onsi/ginkgo/types"
)

func BuildRunCommand() *Command {
	commandFlags := NewRunCommandFlags(flag.NewFlagSet("ginkgo", flag.ExitOnError))
	notifier := NewNotifier(commandFlags)
	interruptHandler := interrupthandler.NewInterruptHandler()
	runner := &SpecRunner{
		commandFlags:     commandFlags,
		notifier:         notifier,
		interruptHandler: interruptHandler,
		suiteRunner:      NewSuiteRunner(notifier, interruptHandler),
	}

	return &Command{
		Name:         "",
		FlagSet:      commandFlags.FlagSet,
		UsageCommand: "ginkgo <FLAGS> <PACKAGES> -- <PASS-THROUGHS>",
		Usage: []string{
			"Run the tests in the passed in <PACKAGES> (or the package in the current directory if left blank).",
			"Any arguments after -- will be passed to the test.",
			"Accepts the following flags:",
		},
		Command: runner.RunSpecs,
	}
}

type SpecRunner struct {
	commandFlags     *RunWatchAndBuildCommandFlags
	notifier         *Notifier
	interruptHandler *interrupthandler.InterruptHandler
	suiteRunner      *SuiteRunner
}

func (r *SpecRunner) RunSpecs(args []string, additionalArgs []string) {
	r.commandFlags.computeNodes()
	r.notifier.VerifyNotificationsAreAvailable()

	suites, skippedPackages := findSuites(args, r.commandFlags.Recurse, r.commandFlags.SkipPackage, true)
	if len(skippedPackages) > 0 {
		fmt.Println("Will skip:")
		for _, skippedPackage := range skippedPackages {
			fmt.Println("  " + skippedPackage)
		}
	}

	if len(skippedPackages) > 0 && len(suites) == 0 {
		fmt.Println("All tests skipped!  Exiting...")
		os.Exit(0)
	}

	if len(suites) == 0 {
		complainAndQuit("Found no test suites")
	}

	r.ComputeSuccinctMode(len(suites))

	t := time.Now()

	runners := []*testrunner.TestRunner{}
	for _, suite := range suites {
		runners = append(runners, testrunner.New(suite, r.commandFlags.NumCPU, r.commandFlags.ParallelStream, r.commandFlags.GoOpts, additionalArgs))
	}

	numSuites := 0
	runResult := testrunner.PassingRunResult()
	if r.commandFlags.UntilItFails {
		iteration := 0
		for {
			r.UpdateSeed()
			randomizedRunners := r.randomizeOrder(runners)
			runResult, numSuites = r.suiteRunner.RunSuites(randomizedRunners, r.commandFlags.NumCompilers, r.commandFlags.KeepGoing, nil)
			iteration++

			if r.interruptHandler.WasInterrupted() {
				break
			}

			if runResult.Passed {
				fmt.Printf("\nAll tests passed...\nWill keep running them until they fail.\nThis was attempt #%d\n%s\n", iteration, orcMessage(iteration))
			} else {
				fmt.Printf("\nTests failed on attempt #%d\n\n", iteration)
				break
			}
		}
	} else {
		randomizedRunners := r.randomizeOrder(runners)
		runResult, numSuites = r.suiteRunner.RunSuites(randomizedRunners, r.commandFlags.NumCompilers, r.commandFlags.KeepGoing, nil)
	}

	for _, runner := range runners {
		runner.CleanUp()
	}

	fmt.Printf("\nGinkgo ran %d %s in %s\n", numSuites, pluralizedWord("suite", "suites", numSuites), time.Since(t))

	if runResult.Passed {
		if runResult.HasProgrammaticFocus {
			fmt.Printf("Test Suite Passed\n")
			fmt.Printf("Detected Programmatic Focus - setting exit status to %d\n", types.GINKGO_FOCUS_EXIT_CODE)
			os.Exit(types.GINKGO_FOCUS_EXIT_CODE)
		} else {
			fmt.Printf("Test Suite Passed\n")
			os.Exit(0)
		}
	} else {
		fmt.Printf("Test Suite Failed\n")
		os.Exit(1)
	}
}

func (r *SpecRunner) ComputeSuccinctMode(numSuites int) {
	if config.DefaultReporterConfig.Verbose {
		config.DefaultReporterConfig.Succinct = false
		return
	}

	if numSuites == 1 {
		return
	}

	if numSuites > 1 && !r.commandFlags.wasSet("succinct") {
		config.DefaultReporterConfig.Succinct = true
	}
}

func (r *SpecRunner) UpdateSeed() {
	if !r.commandFlags.wasSet("seed") {
		config.GinkgoConfig.RandomSeed = time.Now().Unix()
	}
}

func (r *SpecRunner) randomizeOrder(runners []*testrunner.TestRunner) []*testrunner.TestRunner {
	if !r.commandFlags.RandomizeSuites {
		return runners
	}

	if len(runners) <= 1 {
		return runners
	}

	randomizedRunners := make([]*testrunner.TestRunner, len(runners))
	randomizer := rand.New(rand.NewSource(config.GinkgoConfig.RandomSeed))
	permutation := randomizer.Perm(len(runners))
	for i, j := range permutation {
		randomizedRunners[i] = runners[j]
	}
	return randomizedRunners
}

func orcMessage(iteration int) string {
	if iteration < 10 {
		return ""
	} else if iteration < 30 {
		return []string{
			"If at first you succeed...",
			"...try, try again.",
			"Looking good!",
			"Still good...",
			"I think your tests are fine....",
			"Yep, still passing",
			"Here we go again...",
			"Even the gophers are getting bored",
			"Did you try -race?",
			"Maybe you should stop now?",
			"I'm getting tired...",
			"What if I just made you a sandwich?",
			"Hit ^C, hit ^C, please hit ^C",
			"Make it stop. Please!",
			"Come on!  Enough is enough!",
			"Dave, this conversation can serve no purpose anymore. Goodbye.",
			"Just what do you think you're doing, Dave? ",
			"I, Sisyphus",
			"Insanity: doing the same thing over and over again and expecting different results. -Einstein",
			"I guess Einstein never tried to churn butter",
		}[iteration-10] + "\n"
	} else {
		return "No, seriously... you can probably stop now.\n"
	}
}
