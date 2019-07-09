package godog

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/DATA-DOG/godog/colors"
)

const (
	exitSuccess int = iota
	exitFailure
	exitOptionError
)

type initializer func(*Suite)

type runner struct {
	randomSeed            int64
	stopOnFailure, strict bool
	features              []*feature
	fmt                   Formatter
	initializer           initializer
}

func (r *runner) concurrent(rate int) (failed bool) {
	queue := make(chan int, rate)
	for i, ft := range r.features {
		queue <- i // reserve space in queue
		go func(fail *bool, feat *feature) {
			defer func() {
				<-queue // free a space in queue
			}()
			if r.stopOnFailure && *fail {
				return
			}
			suite := &Suite{
				fmt:           r.fmt,
				randomSeed:    r.randomSeed,
				stopOnFailure: r.stopOnFailure,
				strict:        r.strict,
				features:      []*feature{feat},
			}
			r.initializer(suite)
			suite.run()
			if suite.failed {
				*fail = true
			}
		}(&failed, ft)
	}
	// wait until last are processed
	for i := 0; i < rate; i++ {
		queue <- i
	}
	close(queue)

	// print summary
	r.fmt.Summary()
	return
}

func (r *runner) run() bool {
	suite := &Suite{
		fmt:           r.fmt,
		randomSeed:    r.randomSeed,
		stopOnFailure: r.stopOnFailure,
		strict:        r.strict,
		features:      r.features,
	}
	r.initializer(suite)
	suite.run()

	r.fmt.Summary()
	return suite.failed
}

// RunWithOptions is same as Run function, except
// it uses Options provided in order to run the
// test suite without parsing flags
//
// This method is useful in case if you run
// godog in for example TestMain function together
// with go tests
//
// The exit codes may vary from:
//  0 - success
//  1 - failed
//  2 - command line usage error
//  128 - or higher, os signal related error exit codes
//
// If there are flag related errors they will
// be directed to os.Stderr
func RunWithOptions(suite string, contextInitializer func(suite *Suite), opt Options) int {
	var output io.Writer = os.Stdout
	if nil != opt.Output {
		output = opt.Output
	}

	if opt.NoColors {
		output = colors.Uncolored(output)
	} else {
		output = colors.Colored(output)
	}

	if opt.ShowStepDefinitions {
		s := &Suite{}
		contextInitializer(s)
		s.printStepDefinitions(output)
		return exitOptionError
	}

	if len(opt.Paths) == 0 {
		inf, err := os.Stat("features")
		if err == nil && inf.IsDir() {
			opt.Paths = []string{"features"}
		}
	}

	if opt.Concurrency > 1 && !supportsConcurrency(opt.Format) {
		fmt.Fprintln(os.Stderr, fmt.Errorf("format \"%s\" does not support concurrent execution", opt.Format))
		return exitOptionError
	}
	formatter := FindFmt(opt.Format)
	if nil == formatter {
		var names []string
		for name := range AvailableFormatters() {
			names = append(names, name)
		}
		fmt.Fprintln(os.Stderr, fmt.Errorf(
			`unregistered formatter name: "%s", use one of: %s`,
			opt.Format,
			strings.Join(names, ", "),
		))
		return exitOptionError
	}

	features, err := parseFeatures(opt.Tags, opt.Paths)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return exitOptionError
	}

	// user may have specified -1 option to create random seed
	randomize := opt.Randomize
	if randomize == -1 {
		randomize = makeRandomSeed()
	}

	r := runner{
		fmt:           formatter(suite, output),
		initializer:   contextInitializer,
		features:      features,
		randomSeed:    randomize,
		stopOnFailure: opt.StopOnFailure,
		strict:        opt.Strict,
	}

	// store chosen seed in environment, so it could be seen in formatter summary report
	os.Setenv("GODOG_SEED", strconv.FormatInt(r.randomSeed, 10))
	// determine tested package
	_, filename, _, _ := runtime.Caller(1)
	os.Setenv("GODOG_TESTED_PACKAGE", runsFromPackage(filename))

	var failed bool
	if opt.Concurrency > 1 {
		failed = r.concurrent(opt.Concurrency)
	} else {
		failed = r.run()
	}

	// @TODO: should prevent from having these
	os.Setenv("GODOG_SEED", "")
	os.Setenv("GODOG_TESTED_PACKAGE", "")
	if failed && opt.Format != "events" {
		return exitFailure
	}
	return exitSuccess
}

func runsFromPackage(fp string) string {
	dir := filepath.Dir(fp)
	for _, gp := range gopaths {
		gp = filepath.Join(gp, "src")
		if strings.Index(dir, gp) == 0 {
			return strings.TrimLeft(strings.Replace(dir, gp, "", 1), string(filepath.Separator))
		}
	}
	return dir
}

// Run creates and runs the feature suite.
// Reads all configuration options from flags.
// uses contextInitializer to register contexts
//
// the concurrency option allows runner to
// initialize a number of suites to be run
// separately. Only progress formatter
// is supported when concurrency level is
// higher than 1
//
// contextInitializer must be able to register
// the step definitions and event handlers.
//
// The exit codes may vary from:
//  0 - success
//  1 - failed
//  2 - command line usage error
//  128 - or higher, os signal related error exit codes
//
// If there are flag related errors they will
// be directed to os.Stderr
func Run(suite string, contextInitializer func(suite *Suite)) int {
	var opt Options
	opt.Output = colors.Colored(os.Stdout)
	flagSet := FlagSet(&opt)
	if err := flagSet.Parse(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return exitOptionError
	}

	opt.Paths = flagSet.Args()

	return RunWithOptions(suite, contextInitializer, opt)
}

func supportsConcurrency(format string) bool {
	switch format {
	case "events":
	case "junit":
	case "pretty":
	case "cucumber":
	default:
		return true // supports concurrency
	}

	return false // does not support concurrency
}
