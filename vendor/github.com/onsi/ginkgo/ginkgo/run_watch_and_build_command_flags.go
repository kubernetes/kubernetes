package main

import (
	"flag"
	"runtime"

	"time"

	"github.com/onsi/ginkgo/config"
)

type RunWatchAndBuildCommandFlags struct {
	Recurse     bool
	SkipPackage string
	GoOpts      map[string]interface{}

	//for run and watch commands
	NumCPU         int
	NumCompilers   int
	ParallelStream bool
	Notify         bool
	AfterSuiteHook string
	AutoNodes      bool
	Timeout        time.Duration

	//only for run command
	KeepGoing       bool
	UntilItFails    bool
	RandomizeSuites bool

	//only for watch command
	Depth       int
	WatchRegExp string

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

func (c *RunWatchAndBuildCommandFlags) stringSlot(slot string) *string {
	var opt string
	c.GoOpts[slot] = &opt
	return &opt
}

func (c *RunWatchAndBuildCommandFlags) boolSlot(slot string) *bool {
	var opt bool
	c.GoOpts[slot] = &opt
	return &opt
}

func (c *RunWatchAndBuildCommandFlags) intSlot(slot string) *int {
	var opt int
	c.GoOpts[slot] = &opt
	return &opt
}

func (c *RunWatchAndBuildCommandFlags) flags(mode int) {
	c.GoOpts = make(map[string]interface{})

	onWindows := (runtime.GOOS == "windows")

	c.FlagSet.BoolVar(&(c.Recurse), "r", false, "Find and run test suites under the current directory recursively.")
	c.FlagSet.BoolVar(c.boolSlot("race"), "race", false, "Run tests with race detection enabled.")
	c.FlagSet.BoolVar(c.boolSlot("cover"), "cover", false, "Run tests with coverage analysis, will generate coverage profiles with the package name in the current directory.")
	c.FlagSet.StringVar(c.stringSlot("coverpkg"), "coverpkg", "", "Run tests with coverage on the given external modules.")
	c.FlagSet.StringVar(&(c.SkipPackage), "skipPackage", "", "A comma-separated list of package names to be skipped.  If any part of the package's path matches, that package is ignored.")
	c.FlagSet.StringVar(c.stringSlot("tags"), "tags", "", "A list of build tags to consider satisfied during the build.")
	c.FlagSet.StringVar(c.stringSlot("gcflags"), "gcflags", "", "Arguments to pass on each go tool compile invocation.")
	c.FlagSet.StringVar(c.stringSlot("covermode"), "covermode", "", "Set the mode for coverage analysis.")
	c.FlagSet.BoolVar(c.boolSlot("a"), "a", false, "Force rebuilding of packages that are already up-to-date.")
	c.FlagSet.BoolVar(c.boolSlot("n"), "n", false, "Have `go test` print the commands but do not run them.")
	c.FlagSet.BoolVar(c.boolSlot("msan"), "msan", false, "Enable interoperation with memory sanitizer.")
	c.FlagSet.BoolVar(c.boolSlot("x"), "x", false, "Have `go test` print the commands.")
	c.FlagSet.BoolVar(c.boolSlot("work"), "work", false, "Print the name of the temporary work directory and do not delete it when exiting.")
	c.FlagSet.StringVar(c.stringSlot("asmflags"), "asmflags", "", "Arguments to pass on each go tool asm invocation.")
	c.FlagSet.StringVar(c.stringSlot("buildmode"), "buildmode", "", "Build mode to use. See 'go help buildmode' for more.")
	c.FlagSet.StringVar(c.stringSlot("mod"), "mod", "", "Go module control. See 'go help modules' for more.")
	c.FlagSet.StringVar(c.stringSlot("compiler"), "compiler", "", "Name of compiler to use, as in runtime.Compiler (gccgo or gc).")
	c.FlagSet.StringVar(c.stringSlot("gccgoflags"), "gccgoflags", "", "Arguments to pass on each gccgo compiler/linker invocation.")
	c.FlagSet.StringVar(c.stringSlot("installsuffix"), "installsuffix", "", "A suffix to use in the name of the package installation directory.")
	c.FlagSet.StringVar(c.stringSlot("ldflags"), "ldflags", "", "Arguments to pass on each go tool link invocation.")
	c.FlagSet.BoolVar(c.boolSlot("linkshared"), "linkshared", false, "Link against shared libraries previously created with -buildmode=shared.")
	c.FlagSet.StringVar(c.stringSlot("pkgdir"), "pkgdir", "", "install and load all packages from the given dir instead of the usual locations.")
	c.FlagSet.StringVar(c.stringSlot("toolexec"), "toolexec", "", "a program to use to invoke toolchain programs like vet and asm.")
	c.FlagSet.IntVar(c.intSlot("blockprofilerate"), "blockprofilerate", 1, "Control the detail provided in goroutine blocking profiles by calling runtime.SetBlockProfileRate with the given value.")
	c.FlagSet.StringVar(c.stringSlot("coverprofile"), "coverprofile", "", "Write a coverage profile to the specified file after all tests have passed.")
	c.FlagSet.StringVar(c.stringSlot("cpuprofile"), "cpuprofile", "", "Write a CPU profile to the specified file before exiting.")
	c.FlagSet.StringVar(c.stringSlot("memprofile"), "memprofile", "", "Write a memory profile to the specified file after all tests have passed.")
	c.FlagSet.IntVar(c.intSlot("memprofilerate"), "memprofilerate", 0, "Enable more precise (and expensive) memory profiles by setting runtime.MemProfileRate.")
	c.FlagSet.StringVar(c.stringSlot("outputdir"), "outputdir", "", "Place output files from profiling in the specified directory.")
	c.FlagSet.BoolVar(c.boolSlot("requireSuite"), "requireSuite", false, "Fail if there are ginkgo tests in a directory but no test suite (missing RunSpecs)")
	c.FlagSet.StringVar(c.stringSlot("vet"), "vet", "", "Configure the invocation of 'go vet' to use the comma-separated list of vet checks. If list is 'off', 'go test' does not run 'go vet' at all.")

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
		c.FlagSet.DurationVar(&(c.Timeout), "timeout", 24*time.Hour, "Suite fails if it does not complete within the specified timeout")
	}

	if mode == runMode {
		c.FlagSet.BoolVar(&(c.KeepGoing), "keepGoing", false, "When true, failures from earlier test suites do not prevent later test suites from running")
		c.FlagSet.BoolVar(&(c.UntilItFails), "untilItFails", false, "When true, Ginkgo will keep rerunning tests until a failure occurs")
		c.FlagSet.BoolVar(&(c.RandomizeSuites), "randomizeSuites", false, "When true, Ginkgo will randomize the order in which test suites run")
	}

	if mode == watchMode {
		c.FlagSet.IntVar(&(c.Depth), "depth", 1, "Ginkgo will watch dependencies down to this depth in the dependency tree")
		c.FlagSet.StringVar(&(c.WatchRegExp), "watchRegExp", `\.go$`, "Files matching this regular expression will be watched for changes")
	}
}
