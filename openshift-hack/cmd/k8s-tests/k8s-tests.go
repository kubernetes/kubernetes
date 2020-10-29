package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	utilflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/logs"
	"k8s.io/kubernetes/test/e2e/framework"

	// initialize framework extensions
	_ "k8s.io/kubernetes/test/e2e/framework/debug/init"
	_ "k8s.io/kubernetes/test/e2e/framework/metrics/init"
)

func main() {
	logs.InitLogs()
	defer logs.FlushLogs()

	rand.Seed(time.Now().UTC().UnixNano())

	pflag.CommandLine.SetNormalizeFunc(utilflag.WordSepNormalizeFunc)

	root := &cobra.Command{
		Long: "OpenShift Tests compatible wrapper",
	}

	root.AddCommand(
		newRunTestCommand(),
		newListTestsCommand(),
	)

	f := flag.CommandLine.Lookup("v")
	root.PersistentFlags().AddGoFlag(f)
	pflag.CommandLine = pflag.NewFlagSet("empty", pflag.ExitOnError)
	flag.CommandLine = flag.NewFlagSet("empty", flag.ExitOnError)
	framework.RegisterCommonFlags(flag.CommandLine)
	framework.RegisterClusterFlags(flag.CommandLine)

	if err := func() error {
		return root.Execute()
	}(); err != nil {
		if ex, ok := err.(ExitError); ok {
			fmt.Fprintf(os.Stderr, "Ginkgo exit error %d: %v\n", ex.Code, err)
			os.Exit(ex.Code)
		}
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func newRunTestCommand() *cobra.Command {
	testOpt := NewTestOptions(os.Stdout, os.Stderr)

	cmd := &cobra.Command{
		Use:          "run-test NAME",
		Short:        "Run a single test by name",
		Long:         "Execute a single test.",
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			if err := initializeTestFramework(os.Getenv("TEST_PROVIDER")); err != nil {
				return err
			}

			return testOpt.Run(args)
		},
	}
	return cmd
}

func newListTestsCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:          "list",
		Short:        "List available tests",
		Long:         "List the available tests in this binary.",
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			tests := testsForSuite()
			sort.Slice(tests, func(i, j int) bool { return tests[i].Name < tests[j].Name })
			data, err := json.Marshal(tests)
			if err != nil {
				return err
			}
			fmt.Fprintf(os.Stdout, "%s\n", data)
			return nil
		},
	}

	return cmd
}
