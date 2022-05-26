package slowest

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"time"

	"github.com/dnephin/pflag"
	"gotest.tools/gotestsum/internal/aggregate"
	"gotest.tools/gotestsum/log"
	"gotest.tools/gotestsum/testjson"
)

// Run the command
func Run(name string, args []string) error {
	flags, opts := setupFlags(name)
	switch err := flags.Parse(args); {
	case err == pflag.ErrHelp:
		return nil
	case err != nil:
		usage(os.Stderr, name, flags)
		return err
	}
	return run(opts)
}

func setupFlags(name string) (*pflag.FlagSet, *options) {
	opts := &options{}
	flags := pflag.NewFlagSet(name, pflag.ContinueOnError)
	flags.SetInterspersed(false)
	flags.Usage = func() {
		usage(os.Stdout, name, flags)
	}
	flags.StringVar(&opts.jsonfile, "jsonfile", os.Getenv("GOTESTSUM_JSONFILE"),
		"path to test2json output, defaults to stdin")
	flags.DurationVar(&opts.threshold, "threshold", 100*time.Millisecond,
		"test cases with elapsed time greater than threshold are slow tests")
	flags.StringVar(&opts.skipStatement, "skip-stmt", "",
		"add this go statement to slow tests, instead of printing the list of slow tests")
	flags.BoolVar(&opts.debug, "debug", false,
		"enable debug logging.")
	return flags, opts
}

func usage(out io.Writer, name string, flags *pflag.FlagSet) {
	fmt.Fprintf(out, `Usage:
    %[1]s [flags]

Read a json file and print or update tests which are slower than threshold.
The json file may be created with 'gotestsum --jsonfile' or 'go test -json'.
If a TestCase appears more than once in the json file, it will only appear once
in the output, and the median value of all the elapsed times will be used.

By default this command will print the list of tests slower than threshold to stdout.
The list will be sorted from slowest to fastest.

If --skip-stmt is set, instead of printing the list to stdout, the AST for the
Go source code in the working directory tree will be modified. The value of
--skip-stmt will be added to Go test files as the first statement in all the test
functions which are slower than threshold.

The --skip-stmt flag may be set to the name of a predefined statement, or to
Go source code which will be parsed as a go/ast.Stmt. Currently there is only one
predefined statement, --skip-stmt=testing.Short, which uses this Go statement:

    if testing.Short() {
        t.Skip("too slow for testing.Short")
    }


Alternatively, a custom --skip-stmt may be provided as a string:

    skip_stmt='
        if os.GetEnv("TEST_FAST") != "" {
            t.Skip("too slow for TEST_FAST")
        }
    '
    go test -json -short ./... | %[1]s --skip-stmt "$skip_stmt"

Note that this tool does not add imports, so using a custom statement may require
you to add imports to the file.

Go build flags, such as build tags, may be set using the GOFLAGS environment
variable, following the same rules as the go toolchain. See
https://golang.org/cmd/go/#hdr-Environment_variables.

Flags:
`, name)
	flags.SetOutput(out)
	flags.PrintDefaults()
}

type options struct {
	threshold     time.Duration
	jsonfile      string
	skipStatement string
	debug         bool
}

func run(opts *options) error {
	if opts.debug {
		log.SetLevel(log.DebugLevel)
	}
	in, err := jsonfileReader(opts.jsonfile)
	if err != nil {
		return fmt.Errorf("failed to read jsonfile: %v", err)
	}
	defer func() {
		if err := in.Close(); err != nil {
			log.Errorf("Failed to close file %v: %v", opts.jsonfile, err)
		}
	}()

	exec, err := testjson.ScanTestOutput(testjson.ScanConfig{Stdout: in})
	if err != nil {
		return fmt.Errorf("failed to scan testjson: %v", err)
	}

	tcs := aggregate.Slowest(exec, opts.threshold)
	if opts.skipStatement != "" {
		skipStmt, err := parseSkipStatement(opts.skipStatement)
		if err != nil {
			return fmt.Errorf("failed to parse skip expr: %v", err)
		}
		return writeTestSkip(tcs, skipStmt)
	}
	for _, tc := range tcs {
		fmt.Printf("%s %s %v\n", tc.Package, tc.Test, tc.Elapsed)
	}

	return nil
}

func jsonfileReader(v string) (io.ReadCloser, error) {
	switch v {
	case "", "-":
		return ioutil.NopCloser(os.Stdin), nil
	default:
		return os.Open(v)
	}
}
