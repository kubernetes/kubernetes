package testjson

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	"github.com/jonboulle/clockwork"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/sync/errgroup"
)

// Action of TestEvent
type Action string

// nolint: unused
const (
	ActionRun    Action = "run"
	ActionPause  Action = "pause"
	ActionCont   Action = "cont"
	ActionPass   Action = "pass"
	ActionBench  Action = "bench"
	ActionFail   Action = "fail"
	ActionOutput Action = "output"
	ActionSkip   Action = "skip"
)

// TestEvent is a structure output by go tool test2json and go test -json.
type TestEvent struct {
	// Time encoded as an RFC3339-format string
	Time    time.Time
	Action  Action
	Package string
	Test    string
	// Elapsed time in seconds
	Elapsed float64
	// Output of test or benchmark
	Output string
	// raw is the raw JSON bytes of the event
	raw []byte
}

// PackageEvent returns true if the event is a package start or end event
func (e TestEvent) PackageEvent() bool {
	return e.Test == ""
}

// ElapsedFormatted returns Elapsed formatted in the go test format, ex (0.00s).
func (e TestEvent) ElapsedFormatted() string {
	return fmt.Sprintf("(%.2fs)", e.Elapsed)
}

// Bytes returns the serialized JSON bytes that were parsed to create the event.
func (e TestEvent) Bytes() []byte {
	return e.raw
}

// Package is the set of TestEvents for a single go package
type Package struct {
	// TODO: this could be Total()
	Total   int
	Failed  []TestCase
	Skipped []TestCase
	Passed  []TestCase
	output  map[string][]string
	// coverage stores the code coverage output for the package without the
	// trailing newline (ex: coverage: 91.1% of statements).
	coverage string
	// action identifies if the package passed or failed. A package may fail
	// with no test failures if an init() or TestMain exits non-zero.
	// skip indicates there were no tests.
	action Action
}

// Result returns if the package passed, failed, or was skipped because there
// were no tests.
func (p Package) Result() Action {
	return p.action
}

// Elapsed returns the sum of the elapsed time for all tests in the package.
func (p Package) Elapsed() time.Duration {
	elapsed := time.Duration(0)
	for _, testcase := range p.TestCases() {
		elapsed += testcase.Elapsed
	}
	return elapsed
}

// TestCases returns all the test cases.
func (p Package) TestCases() []TestCase {
	return append(append(p.Passed, p.Failed...), p.Skipped...)
}

// Output returns the full test output for a test.
func (p Package) Output(test string) string {
	return strings.Join(p.output[test], "")
}

// TestMainFailed returns true if the package failed, but there were no tests.
// This may occur if the package init() or TestMain exited non-zero.
func (p Package) TestMainFailed() bool {
	return p.action == ActionFail && len(p.Failed) == 0
}

// TestCase stores the name and elapsed time for a test case.
type TestCase struct {
	Package string
	Test    string
	Elapsed time.Duration
}

func newPackage() *Package {
	return &Package{output: make(map[string][]string)}
}

// Execution of one or more test packages
type Execution struct {
	started  time.Time
	packages map[string]*Package
	errors   []string
}

func (e *Execution) add(event TestEvent) {
	pkg, ok := e.packages[event.Package]
	if !ok {
		pkg = newPackage()
		e.packages[event.Package] = pkg
	}
	if event.PackageEvent() {
		e.addPackageEvent(pkg, event)
		return
	}
	e.addTestEvent(pkg, event)
}

func (e *Execution) addPackageEvent(pkg *Package, event TestEvent) {
	switch event.Action {
	case ActionPass, ActionFail:
		pkg.action = event.Action
	case ActionOutput:
		if isCoverageOutput(event.Output) {
			pkg.coverage = strings.TrimRight(event.Output, "\n")
		}
		pkg.output[""] = append(pkg.output[""], event.Output)
	}
}

func (e *Execution) addTestEvent(pkg *Package, event TestEvent) {
	switch event.Action {
	case ActionRun:
		pkg.Total++
	case ActionFail:
		pkg.Failed = append(pkg.Failed, TestCase{
			Package: event.Package,
			Test:    event.Test,
			Elapsed: elapsedDuration(event.Elapsed),
		})
	case ActionSkip:
		pkg.Skipped = append(pkg.Skipped, TestCase{
			Package: event.Package,
			Test:    event.Test,
			Elapsed: elapsedDuration(event.Elapsed),
		})
	case ActionOutput, ActionBench:
		// TODO: limit size of buffered test output
		pkg.output[event.Test] = append(pkg.output[event.Test], event.Output)
	case ActionPass:
		pkg.Passed = append(pkg.Passed, TestCase{
			Package: event.Package,
			Test:    event.Test,
			Elapsed: elapsedDuration(event.Elapsed),
		})
		// Remove test output once a test passes, it wont be used
		delete(pkg.output, event.Test)
	}
}

func elapsedDuration(elapsed float64) time.Duration {
	return time.Duration(elapsed*1000) * time.Millisecond
}

func isCoverageOutput(output string) bool {
	return all(
		strings.HasPrefix(output, "coverage:"),
		strings.HasSuffix(output, "% of statements\n"))
}

// Output returns the full test output for a test.
func (e *Execution) Output(pkg, test string) string {
	return strings.Join(e.packages[pkg].output[test], "")
}

// OutputLines returns the full test output for a test as an array of lines.
func (e *Execution) OutputLines(pkg, test string) []string {
	return e.packages[pkg].output[test]
}

// Package returns the Package by name.
func (e *Execution) Package(name string) *Package {
	return e.packages[name]
}

// Packages returns a sorted list of all package names.
func (e *Execution) Packages() []string {
	return sortedKeys(e.packages)
}

var clock = clockwork.NewRealClock()

// Elapsed returns the time elapsed since the execution started.
func (e *Execution) Elapsed() time.Duration {
	return clock.Now().Sub(e.started)
}

// Failed returns a list of all the failed test cases.
func (e *Execution) Failed() []TestCase {
	var failed []TestCase
	for _, name := range sortedKeys(e.packages) {
		pkg := e.packages[name]

		// Add package-level failure output if there were no failed tests.
		if pkg.TestMainFailed() {
			failed = append(failed, TestCase{Package: name})
		} else {
			failed = append(failed, pkg.Failed...)
		}
	}
	return failed
}

func sortedKeys(pkgs map[string]*Package) []string {
	keys := make([]string, 0, len(pkgs))
	for key := range pkgs {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// Skipped returns a list of all the skipped test cases.
func (e *Execution) Skipped() []TestCase {
	skipped := make([]TestCase, 0, len(e.packages))
	for _, pkg := range sortedKeys(e.packages) {
		skipped = append(skipped, e.packages[pkg].Skipped...)
	}
	return skipped
}

// Total returns a count of all test cases.
func (e *Execution) Total() int {
	total := 0
	for _, pkg := range e.packages {
		total += pkg.Total
	}
	return total
}

func (e *Execution) addError(err string) {
	// Build errors start with a header
	if strings.HasPrefix(err, "# ") {
		return
	}
	// TODO: may need locking, or use a channel
	e.errors = append(e.errors, err)
}

// Errors returns a list of all the errors.
func (e *Execution) Errors() []string {
	return e.errors
}

// NewExecution returns a new Execution and records the current time as the
// time the test execution started.
func NewExecution() *Execution {
	return &Execution{
		started:  time.Now(),
		packages: make(map[string]*Package),
	}
}

// ScanConfig used by ScanTestOutput
type ScanConfig struct {
	Stdout  io.Reader
	Stderr  io.Reader
	Handler EventHandler
}

// EventHandler is called by ScanTestOutput for each event and write to stderr.
type EventHandler interface {
	Event(event TestEvent, execution *Execution) error
	Err(text string) error
}

// ScanTestOutput reads lines from stdout and stderr, creates an Execution,
// calls the Handler for each event, and returns the Execution.
func ScanTestOutput(config ScanConfig) (*Execution, error) {
	execution := NewExecution()
	var group errgroup.Group
	group.Go(func() error {
		return readStdout(config, execution)
	})
	group.Go(func() error {
		return readStderr(config, execution)
	})
	return execution, group.Wait()
}

func readStdout(config ScanConfig, execution *Execution) error {
	scanner := bufio.NewScanner(config.Stdout)
	for scanner.Scan() {
		raw := scanner.Bytes()
		event, err := parseEvent(raw)
		switch {
		case err == errBadEvent:
			// nolint: errcheck
			config.Handler.Err(errBadEvent.Error() + ": " + scanner.Text())
			continue
		case err != nil:
			return errors.Wrapf(err, "failed to parse test output: %s", string(raw))
		}
		execution.add(event)
		if err := config.Handler.Event(event, execution); err != nil {
			return err
		}
	}
	return errors.Wrap(scanner.Err(), "failed to scan test output")
}

func readStderr(config ScanConfig, execution *Execution) error {
	scanner := bufio.NewScanner(config.Stderr)
	for scanner.Scan() {
		line := scanner.Text()
		config.Handler.Err(line) // nolint: errcheck
		if isGoModuleOutput(line) {
			continue
		}
		execution.addError(line)
	}
	return errors.Wrap(scanner.Err(), "failed to scan test stderr")
}

func isGoModuleOutput(scannerText string) bool {
	prefixes := []string{
		"go: copying",
		"go: creating",
		"go: downloading",
		"go: extracting",
		"go: finding",
	}

	for _, prefix := range prefixes {
		if strings.HasPrefix(scannerText, prefix) {
			return true
		}
	}
	return false
}

func parseEvent(raw []byte) (TestEvent, error) {
	// TODO: this seems to be a bug in the `go test -json` output
	if bytes.HasPrefix(raw, []byte("FAIL")) {
		logrus.Warn(string(raw))
		return TestEvent{}, errBadEvent
	}

	event := TestEvent{}
	err := json.Unmarshal(raw, &event)
	event.raw = raw
	return event, err
}

var errBadEvent = errors.New("bad output from test2json")
