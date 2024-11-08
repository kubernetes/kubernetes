package build

import (
	"fmt"
	"os"
	"path"

	"github.com/onsi/ginkgo/v2/ginkgo/command"
	"github.com/onsi/ginkgo/v2/ginkgo/internal"
	"github.com/onsi/ginkgo/v2/types"
)

func BuildBuildCommand() command.Command {
	var cliConfig = types.NewDefaultCLIConfig()
	var goFlagsConfig = types.NewDefaultGoFlagsConfig()

	flags, err := types.BuildBuildCommandFlagSet(&cliConfig, &goFlagsConfig)
	if err != nil {
		panic(err)
	}

	return command.Command{
		Name:     "build",
		Flags:    flags,
		Usage:    "ginkgo build <FLAGS> <PACKAGES>",
		ShortDoc: "Build the passed in <PACKAGES> (or the package in the current directory if left blank).",
		DocLink:  "precompiling-suites",
		Command: func(args []string, _ []string) {
			var errors []error
			cliConfig, goFlagsConfig, errors = types.VetAndInitializeCLIAndGoConfig(cliConfig, goFlagsConfig)
			command.AbortIfErrors("Ginkgo detected configuration issues:", errors)

			buildSpecs(args, cliConfig, goFlagsConfig)
		},
	}
}

func buildSpecs(args []string, cliConfig types.CLIConfig, goFlagsConfig types.GoFlagsConfig) {
	suites := internal.FindSuites(args, cliConfig, false).WithoutState(internal.TestSuiteStateSkippedByFilter)
	if len(suites) == 0 {
		command.AbortWith("Found no test suites")
	}

	internal.VerifyCLIAndFrameworkVersion(suites)

	opc := internal.NewOrderedParallelCompiler(cliConfig.ComputedNumCompilers())
	opc.StartCompiling(suites, goFlagsConfig)

	for {
		suiteIdx, suite := opc.Next()
		if suiteIdx >= len(suites) {
			break
		}
		suites[suiteIdx] = suite
		if suite.State.Is(internal.TestSuiteStateFailedToCompile) {
			fmt.Println(suite.CompilationError.Error())
		} else {
			if len(goFlagsConfig.O) == 0 {
				goFlagsConfig.O = path.Join(suite.Path, suite.PackageName+".test")
			} else {
				stat, err := os.Stat(goFlagsConfig.O)
				if err != nil {
					panic(err)
				}
				if stat.IsDir() {
					goFlagsConfig.O += "/" + suite.PackageName + ".test"
				}
			}
			fmt.Printf("Compiled %s\n", goFlagsConfig.O)
		}
	}

	if suites.CountWithState(internal.TestSuiteStateFailedToCompile) > 0 {
		command.AbortWith("Failed to compile all tests")
	}
}
