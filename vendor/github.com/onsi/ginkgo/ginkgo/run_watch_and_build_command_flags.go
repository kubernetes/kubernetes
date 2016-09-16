package main

import (
	"flag"
	"runtime"

	"github.com/onsi/ginkgo/config"
)

type RunWatchAndBuildCommandFlags struct {
	Recurse     bool
	Race        bool
	Cover       bool
	CoverPkg    string
	SkipPackage string
	Tags        string
	GCFlags     string

	//for run and watch commands
	NumCPU         int
	NumCompilers   int
	ParallelStream bool
	Notify         bool
	AfterSuiteHook string
	AutoNodes      bool

	//only for run command
	KeepGoing       bool
	UntilItFails    bool
	RandomizeSuites bool

	//only for watch command
	Depth int

	FlagSet *flag.FlagSet
}

const runMode = 1
const watchMode = 2
const buildMode = 3

func NewRunCommandFlags(flagSet *flag.FlagSet) *RunWatchAndBuildCommandFlags {
	c := &RunWatchAndBuildCommandFlags{
		FlagSet: flagSet,
	}
	c.flags(runMode)
	return c
}

func NewWatchCommandFlags(flagSet *flag.FlagSet) *RunWatchAndBuildCommandFlags {
	c := &RunWatchAndBuildCommandFlags{
		FlagSet: flagSet,
	}
	c.flags(watchMode)
	return c
}

func NewBuildCommandFlags(flagSet *flag.FlagSet) *RunWatchAndBuildCommandFlags {
	c := &RunWatchAndBuildCommandFlags{
		FlagSet: flagSet,
	}
	c.flags(buildMode)
	return c
}

func (c *RunWatchAndBuildCommandFlags) wasSet(flagName string) bool {
	wasSet := false
	c.FlagSet.Visit(func(f *flag.Flag) {
		if f.Name == flagName {
			wasSet = true
		}
	})

	return wasSet
}

func (c *RunWatchAndBuildCommandFlags) computeNodes() {
	if c.wasSet("nodes") {
		return
	}
	if c.AutoNodes {
		switch n := runtime.NumCPU(); {
		case n <= 4:
			c.NumCPU = n
		default:
			c.NumCPU = n - 1
		}
	}
}

func (c *RunWatchAndBuildCommandFlags) flags(mode int) {
	onWindows := (runtime.GOOS == "windows")

	c.FlagSet.BoolVar(&(c.Recurse), "r", false, "Find and run test suites under the current directory recursively")
	c.FlagSet.BoolVar(&(c.Race), "race", false, "Run tests with race detection enabled")
	c.FlagSet.BoolVar(&(c.Cover), "cover", false, "Run tests with coverage analysis, will generate coverage profiles with the package name in the current directory")
	c.FlagSet.StringVar(&(c.CoverPkg), "coverpkg", "", "Run tests with coverage on the given external modules")
	c.FlagSet.StringVar(&(c.SkipPackage), "skipPackage", "", "A comma-separated list of package names to be skipped.  If any part of the package's path matches, that package is ignored.")
	c.FlagSet.StringVar(&(c.Tags), "tags", "", "A list of build tags to consider satisfied during the build")
	c.FlagSet.StringVar(&(c.GCFlags), "gcflags", "", "Arguments to pass on each go tool compile invocation.")

	if mode == runMode || mode == watchMode {
		config.Flags(c.FlagSet, "", false)
		c.FlagSet.IntVar(&(c.NumCPU), "nodes", 1, "The number of parallel test nodes to run")
		c.FlagSet.IntVar(&(c.NumCompilers), "compilers", 0, "The number of concurrent compilations to run (0 will autodetect)")
		c.FlagSet.BoolVar(&(c.AutoNodes), "p", false, "Run in parallel with auto-detected number of nodes")
		c.FlagSet.BoolVar(&(c.ParallelStream), "stream", onWindows, "stream parallel test output in real time: less coherent, but useful for debugging")
		if !onWindows {
			c.FlagSet.BoolVar(&(c.Notify), "notify", false, "Send desktop notifications when a test run completes")
		}
		c.FlagSet.StringVar(&(c.AfterSuiteHook), "afterSuiteHook", "", "Run a command when a suite test run completes")
	}

	if mode == runMode {
		c.FlagSet.BoolVar(&(c.KeepGoing), "keepGoing", false, "When true, failures from earlier test suites do not prevent later test suites from running")
		c.FlagSet.BoolVar(&(c.UntilItFails), "untilItFails", false, "When true, Ginkgo will keep rerunning tests until a failure occurs")
		c.FlagSet.BoolVar(&(c.RandomizeSuites), "randomizeSuites", false, "When true, Ginkgo will randomize the order in which test suites run")
	}

	if mode == watchMode {
		c.FlagSet.IntVar(&(c.Depth), "depth", 1, "Ginkgo will watch dependencies down to this depth in the dependency tree")
	}
}
