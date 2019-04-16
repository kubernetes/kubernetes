package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/onsi/ginkgo/ginkgo/interrupthandler"
	"github.com/onsi/ginkgo/ginkgo/testrunner"
)

func BuildBuildCommand() *Command {
	commandFlags := NewBuildCommandFlags(flag.NewFlagSet("build", flag.ExitOnError))
	interruptHandler := interrupthandler.NewInterruptHandler()
	builder := &SpecBuilder{
		commandFlags:     commandFlags,
		interruptHandler: interruptHandler,
	}

	return &Command{
		Name:         "build",
		FlagSet:      commandFlags.FlagSet,
		UsageCommand: "ginkgo build <FLAGS> <PACKAGES>",
		Usage: []string{
			"Build the passed in <PACKAGES> (or the package in the current directory if left blank).",
			"Accepts the following flags:",
		},
		Command: builder.BuildSpecs,
	}
}

type SpecBuilder struct {
	commandFlags     *RunWatchAndBuildCommandFlags
	interruptHandler *interrupthandler.InterruptHandler
}

func (r *SpecBuilder) BuildSpecs(args []string, additionalArgs []string) {
	r.commandFlags.computeNodes()

	suites, _ := findSuites(args, r.commandFlags.Recurse, r.commandFlags.SkipPackage, false)

	if len(suites) == 0 {
		complainAndQuit("Found no test suites")
	}

	passed := true
	for _, suite := range suites {
		runner := testrunner.New(suite, 1, false, 0, r.commandFlags.GoOpts, nil)
		fmt.Printf("Compiling %s...\n", suite.PackageName)

		path, _ := filepath.Abs(filepath.Join(suite.Path, fmt.Sprintf("%s.test", suite.PackageName)))
		err := runner.CompileTo(path)
		if err != nil {
			fmt.Println(err.Error())
			passed = false
		} else {
			fmt.Printf("    compiled %s.test\n", suite.PackageName)
		}

		runner.CleanUp()
	}

	if passed {
		os.Exit(0)
	}
	os.Exit(1)
}
