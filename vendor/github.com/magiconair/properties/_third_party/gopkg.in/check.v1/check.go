// Package check is a rich testing extension for Go's testing package.
//
// For details about the project, see:
//
//     http://labix.org/gocheck
//
package check

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// -----------------------------------------------------------------------
// Internal type which deals with suite method calling.

const (
	fixtureKd = iota
	testKd
)

type funcKind int

const (
	succeededSt = iota
	failedSt
	skippedSt
	panickedSt
	fixturePanickedSt
	missedSt
)

type funcStatus int

// A method value can't reach its own Method structure.
type methodType struct {
	reflect.Value
	Info reflect.Method
}

func newMethod(receiver reflect.Value, i int) *methodType {
	return &methodType{receiver.Method(i), receiver.Type().Method(i)}
}

func (method *methodType) PC() uintptr {
	return method.Info.Func.Pointer()
}

func (method *methodType) suiteName() string {
	t := method.Info.Type.In(0)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	return t.Name()
}

func (method *methodType) String() string {
	return method.suiteName() + "." + method.Info.Name
}

func (method *methodType) matches(re *regexp.Regexp) bool {
	return (re.MatchString(method.Info.Name) ||
		re.MatchString(method.suiteName()) ||
		re.MatchString(method.String()))
}

type C struct {
	method    *methodType
	kind      funcKind
	testName  string
	status    funcStatus
	logb      *logger
	logw      io.Writer
	done      chan *C
	reason    string
	mustFail  bool
	tempDir   *tempDir
	benchMem  bool
	startTime time.Time
	timer
}

func (c *C) stopNow() {
	runtime.Goexit()
}

// logger is a concurrency safe byte.Buffer
type logger struct {
	sync.Mutex
	writer bytes.Buffer
}

func (l *logger) Write(buf []byte) (int, error) {
	l.Lock()
	defer l.Unlock()
	return l.writer.Write(buf)
}

func (l *logger) WriteTo(w io.Writer) (int64, error) {
	l.Lock()
	defer l.Unlock()
	return l.writer.WriteTo(w)
}

func (l *logger) String() string {
	l.Lock()
	defer l.Unlock()
	return l.writer.String()
}

// -----------------------------------------------------------------------
// Handling of temporary files and directories.

type tempDir struct {
	sync.Mutex
	path    string
	counter int
}

func (td *tempDir) newPath() string {
	td.Lock()
	defer td.Unlock()
	if td.path == "" {
		var err error
		for i := 0; i != 100; i++ {
			path := fmt.Sprintf("%s%ccheck-%d", os.TempDir(), os.PathSeparator, rand.Int())
			if err = os.Mkdir(path, 0700); err == nil {
				td.path = path
				break
			}
		}
		if td.path == "" {
			panic("Couldn't create temporary directory: " + err.Error())
		}
	}
	result := filepath.Join(td.path, strconv.Itoa(td.counter))
	td.counter += 1
	return result
}

func (td *tempDir) removeAll() {
	td.Lock()
	defer td.Unlock()
	if td.path != "" {
		err := os.RemoveAll(td.path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "WARNING: Error cleaning up temporaries: "+err.Error())
		}
	}
}

// Create a new temporary directory which is automatically removed after
// the suite finishes running.
func (c *C) MkDir() string {
	path := c.tempDir.newPath()
	if err := os.Mkdir(path, 0700); err != nil {
		panic(fmt.Sprintf("Couldn't create temporary directory %s: %s", path, err.Error()))
	}
	return path
}

// -----------------------------------------------------------------------
// Low-level logging functions.

func (c *C) log(args ...interface{}) {
	c.writeLog([]byte(fmt.Sprint(args...) + "\n"))
}

func (c *C) logf(format string, args ...interface{}) {
	c.writeLog([]byte(fmt.Sprintf(format+"\n", args...)))
}

func (c *C) logNewLine() {
	c.writeLog([]byte{'\n'})
}

func (c *C) writeLog(buf []byte) {
	c.logb.Write(buf)
	if c.logw != nil {
		c.logw.Write(buf)
	}
}

func hasStringOrError(x interface{}) (ok bool) {
	_, ok = x.(fmt.Stringer)
	if ok {
		return
	}
	_, ok = x.(error)
	return
}

func (c *C) logValue(label string, value interface{}) {
	if label == "" {
		if hasStringOrError(value) {
			c.logf("... %#v (%q)", value, value)
		} else {
			c.logf("... %#v", value)
		}
	} else if value == nil {
		c.logf("... %s = nil", label)
	} else {
		if hasStringOrError(value) {
			fv := fmt.Sprintf("%#v", value)
			qv := fmt.Sprintf("%q", value)
			if fv != qv {
				c.logf("... %s %s = %s (%s)", label, reflect.TypeOf(value), fv, qv)
				return
			}
		}
		if s, ok := value.(string); ok && isMultiLine(s) {
			c.logf(`... %s %s = "" +`, label, reflect.TypeOf(value))
			c.logMultiLine(s)
		} else {
			c.logf("... %s %s = %#v", label, reflect.TypeOf(value), value)
		}
	}
}

func (c *C) logMultiLine(s string) {
	b := make([]byte, 0, len(s)*2)
	i := 0
	n := len(s)
	for i < n {
		j := i + 1
		for j < n && s[j-1] != '\n' {
			j++
		}
		b = append(b, "...     "...)
		b = strconv.AppendQuote(b, s[i:j])
		if j < n {
			b = append(b, " +"...)
		}
		b = append(b, '\n')
		i = j
	}
	c.writeLog(b)
}

func isMultiLine(s string) bool {
	for i := 0; i+1 < len(s); i++ {
		if s[i] == '\n' {
			return true
		}
	}
	return false
}

func (c *C) logString(issue string) {
	c.log("... ", issue)
}

func (c *C) logCaller(skip int) {
	// This is a bit heavier than it ought to be.
	skip += 1 // Our own frame.
	pc, callerFile, callerLine, ok := runtime.Caller(skip)
	if !ok {
		return
	}
	var testFile string
	var testLine int
	testFunc := runtime.FuncForPC(c.method.PC())
	if runtime.FuncForPC(pc) != testFunc {
		for {
			skip += 1
			if pc, file, line, ok := runtime.Caller(skip); ok {
				// Note that the test line may be different on
				// distinct calls for the same test.  Showing
				// the "internal" line is helpful when debugging.
				if runtime.FuncForPC(pc) == testFunc {
					testFile, testLine = file, line
					break
				}
			} else {
				break
			}
		}
	}
	if testFile != "" && (testFile != callerFile || testLine != callerLine) {
		c.logCode(testFile, testLine)
	}
	c.logCode(callerFile, callerLine)
}

func (c *C) logCode(path string, line int) {
	c.logf("%s:%d:", nicePath(path), line)
	code, err := printLine(path, line)
	if code == "" {
		code = "..." // XXX Open the file and take the raw line.
		if err != nil {
			code += err.Error()
		}
	}
	c.log(indent(code, "    "))
}

var valueGo = filepath.Join("reflect", "value.go")
var asmGo = filepath.Join("runtime", "asm_")

func (c *C) logPanic(skip int, value interface{}) {
	skip++ // Our own frame.
	initialSkip := skip
	for ; ; skip++ {
		if pc, file, line, ok := runtime.Caller(skip); ok {
			if skip == initialSkip {
				c.logf("... Panic: %s (PC=0x%X)\n", value, pc)
			}
			name := niceFuncName(pc)
			path := nicePath(file)
			if strings.Contains(path, "/gopkg.in/check.v") {
				continue
			}
			if name == "Value.call" && strings.HasSuffix(path, valueGo) {
				continue
			}
			if name == "call16" && strings.Contains(path, asmGo) {
				continue
			}
			c.logf("%s:%d\n  in %s", nicePath(file), line, name)
		} else {
			break
		}
	}
}

func (c *C) logSoftPanic(issue string) {
	c.log("... Panic: ", issue)
}

func (c *C) logArgPanic(method *methodType, expectedType string) {
	c.logf("... Panic: %s argument should be %s",
		niceFuncName(method.PC()), expectedType)
}

// -----------------------------------------------------------------------
// Some simple formatting helpers.

var initWD, initWDErr = os.Getwd()

func init() {
	if initWDErr == nil {
		initWD = strings.Replace(initWD, "\\", "/", -1) + "/"
	}
}

func nicePath(path string) string {
	if initWDErr == nil {
		if strings.HasPrefix(path, initWD) {
			return path[len(initWD):]
		}
	}
	return path
}

func niceFuncPath(pc uintptr) string {
	function := runtime.FuncForPC(pc)
	if function != nil {
		filename, line := function.FileLine(pc)
		return fmt.Sprintf("%s:%d", nicePath(filename), line)
	}
	return "<unknown path>"
}

func niceFuncName(pc uintptr) string {
	function := runtime.FuncForPC(pc)
	if function != nil {
		name := path.Base(function.Name())
		if i := strings.Index(name, "."); i > 0 {
			name = name[i+1:]
		}
		if strings.HasPrefix(name, "(*") {
			if i := strings.Index(name, ")"); i > 0 {
				name = name[2:i] + name[i+1:]
			}
		}
		if i := strings.LastIndex(name, ".*"); i != -1 {
			name = name[:i] + "." + name[i+2:]
		}
		if i := strings.LastIndex(name, "·"); i != -1 {
			name = name[:i] + "." + name[i+2:]
		}
		return name
	}
	return "<unknown function>"
}

// -----------------------------------------------------------------------
// Result tracker to aggregate call results.

type Result struct {
	Succeeded        int
	Failed           int
	Skipped          int
	Panicked         int
	FixturePanicked  int
	ExpectedFailures int
	Missed           int    // Not even tried to run, related to a panic in the fixture.
	RunError         error  // Houston, we've got a problem.
	WorkDir          string // If KeepWorkDir is true
}

type resultTracker struct {
	result          Result
	_lastWasProblem bool
	_waiting        int
	_missed         int
	_expectChan     chan *C
	_doneChan       chan *C
	_stopChan       chan bool
}

func newResultTracker() *resultTracker {
	return &resultTracker{_expectChan: make(chan *C), // Synchronous
		_doneChan: make(chan *C, 32), // Asynchronous
		_stopChan: make(chan bool)}   // Synchronous
}

func (tracker *resultTracker) start() {
	go tracker._loopRoutine()
}

func (tracker *resultTracker) waitAndStop() {
	<-tracker._stopChan
}

func (tracker *resultTracker) expectCall(c *C) {
	tracker._expectChan <- c
}

func (tracker *resultTracker) callDone(c *C) {
	tracker._doneChan <- c
}

func (tracker *resultTracker) _loopRoutine() {
	for {
		var c *C
		if tracker._waiting > 0 {
			// Calls still running. Can't stop.
			select {
			// XXX Reindent this (not now to make diff clear)
			case c = <-tracker._expectChan:
				tracker._waiting += 1
			case c = <-tracker._doneChan:
				tracker._waiting -= 1
				switch c.status {
				case succeededSt:
					if c.kind == testKd {
						if c.mustFail {
							tracker.result.ExpectedFailures++
						} else {
							tracker.result.Succeeded++
						}
					}
				case failedSt:
					tracker.result.Failed++
				case panickedSt:
					if c.kind == fixtureKd {
						tracker.result.FixturePanicked++
					} else {
						tracker.result.Panicked++
					}
				case fixturePanickedSt:
					// Track it as missed, since the panic
					// was on the fixture, not on the test.
					tracker.result.Missed++
				case missedSt:
					tracker.result.Missed++
				case skippedSt:
					if c.kind == testKd {
						tracker.result.Skipped++
					}
				}
			}
		} else {
			// No calls.  Can stop, but no done calls here.
			select {
			case tracker._stopChan <- true:
				return
			case c = <-tracker._expectChan:
				tracker._waiting += 1
			case c = <-tracker._doneChan:
				panic("Tracker got an unexpected done call.")
			}
		}
	}
}

// -----------------------------------------------------------------------
// The underlying suite runner.

type suiteRunner struct {
	suite                     interface{}
	setUpSuite, tearDownSuite *methodType
	setUpTest, tearDownTest   *methodType
	tests                     []*methodType
	tracker                   *resultTracker
	tempDir                   *tempDir
	keepDir                   bool
	output                    *outputWriter
	reportedProblemLast       bool
	benchTime                 time.Duration
	benchMem                  bool
}

type RunConf struct {
	Output        io.Writer
	Stream        bool
	Verbose       bool
	Filter        string
	Benchmark     bool
	BenchmarkTime time.Duration // Defaults to 1 second
	BenchmarkMem  bool
	KeepWorkDir   bool
}

// Create a new suiteRunner able to run all methods in the given suite.
func newSuiteRunner(suite interface{}, runConf *RunConf) *suiteRunner {
	var conf RunConf
	if runConf != nil {
		conf = *runConf
	}
	if conf.Output == nil {
		conf.Output = os.Stdout
	}
	if conf.Benchmark {
		conf.Verbose = true
	}

	suiteType := reflect.TypeOf(suite)
	suiteNumMethods := suiteType.NumMethod()
	suiteValue := reflect.ValueOf(suite)

	runner := &suiteRunner{
		suite:     suite,
		output:    newOutputWriter(conf.Output, conf.Stream, conf.Verbose),
		tracker:   newResultTracker(),
		benchTime: conf.BenchmarkTime,
		benchMem:  conf.BenchmarkMem,
		tempDir:   &tempDir{},
		keepDir:   conf.KeepWorkDir,
		tests:     make([]*methodType, 0, suiteNumMethods),
	}
	if runner.benchTime == 0 {
		runner.benchTime = 1 * time.Second
	}

	var filterRegexp *regexp.Regexp
	if conf.Filter != "" {
		if regexp, err := regexp.Compile(conf.Filter); err != nil {
			msg := "Bad filter expression: " + err.Error()
			runner.tracker.result.RunError = errors.New(msg)
			return runner
		} else {
			filterRegexp = regexp
		}
	}

	for i := 0; i != suiteNumMethods; i++ {
		method := newMethod(suiteValue, i)
		switch method.Info.Name {
		case "SetUpSuite":
			runner.setUpSuite = method
		case "TearDownSuite":
			runner.tearDownSuite = method
		case "SetUpTest":
			runner.setUpTest = method
		case "TearDownTest":
			runner.tearDownTest = method
		default:
			prefix := "Test"
			if conf.Benchmark {
				prefix = "Benchmark"
			}
			if !strings.HasPrefix(method.Info.Name, prefix) {
				continue
			}
			if filterRegexp == nil || method.matches(filterRegexp) {
				runner.tests = append(runner.tests, method)
			}
		}
	}
	return runner
}

// Run all methods in the given suite.
func (runner *suiteRunner) run() *Result {
	if runner.tracker.result.RunError == nil && len(runner.tests) > 0 {
		runner.tracker.start()
		if runner.checkFixtureArgs() {
			c := runner.runFixture(runner.setUpSuite, "", nil)
			if c == nil || c.status == succeededSt {
				for i := 0; i != len(runner.tests); i++ {
					c := runner.runTest(runner.tests[i])
					if c.status == fixturePanickedSt {
						runner.skipTests(missedSt, runner.tests[i+1:])
						break
					}
				}
			} else if c != nil && c.status == skippedSt {
				runner.skipTests(skippedSt, runner.tests)
			} else {
				runner.skipTests(missedSt, runner.tests)
			}
			runner.runFixture(runner.tearDownSuite, "", nil)
		} else {
			runner.skipTests(missedSt, runner.tests)
		}
		runner.tracker.waitAndStop()
		if runner.keepDir {
			runner.tracker.result.WorkDir = runner.tempDir.path
		} else {
			runner.tempDir.removeAll()
		}
	}
	return &runner.tracker.result
}

// Create a call object with the given suite method, and fork a
// goroutine with the provided dispatcher for running it.
func (runner *suiteRunner) forkCall(method *methodType, kind funcKind, testName string, logb *logger, dispatcher func(c *C)) *C {
	var logw io.Writer
	if runner.output.Stream {
		logw = runner.output
	}
	if logb == nil {
		logb = new(logger)
	}
	c := &C{
		method:    method,
		kind:      kind,
		testName:  testName,
		logb:      logb,
		logw:      logw,
		tempDir:   runner.tempDir,
		done:      make(chan *C, 1),
		timer:     timer{benchTime: runner.benchTime},
		startTime: time.Now(),
		benchMem:  runner.benchMem,
	}
	runner.tracker.expectCall(c)
	go (func() {
		runner.reportCallStarted(c)
		defer runner.callDone(c)
		dispatcher(c)
	})()
	return c
}

// Same as forkCall(), but wait for call to finish before returning.
func (runner *suiteRunner) runFunc(method *methodType, kind funcKind, testName string, logb *logger, dispatcher func(c *C)) *C {
	c := runner.forkCall(method, kind, testName, logb, dispatcher)
	<-c.done
	return c
}

// Handle a finished call.  If there were any panics, update the call status
// accordingly.  Then, mark the call as done and report to the tracker.
func (runner *suiteRunner) callDone(c *C) {
	value := recover()
	if value != nil {
		switch v := value.(type) {
		case *fixturePanic:
			if v.status == skippedSt {
				c.status = skippedSt
			} else {
				c.logSoftPanic("Fixture has panicked (see related PANIC)")
				c.status = fixturePanickedSt
			}
		default:
			c.logPanic(1, value)
			c.status = panickedSt
		}
	}
	if c.mustFail {
		switch c.status {
		case failedSt:
			c.status = succeededSt
		case succeededSt:
			c.status = failedSt
			c.logString("Error: Test succeeded, but was expected to fail")
			c.logString("Reason: " + c.reason)
		}
	}

	runner.reportCallDone(c)
	c.done <- c
}

// Runs a fixture call synchronously.  The fixture will still be run in a
// goroutine like all suite methods, but this method will not return
// while the fixture goroutine is not done, because the fixture must be
// run in a desired order.
func (runner *suiteRunner) runFixture(method *methodType, testName string, logb *logger) *C {
	if method != nil {
		c := runner.runFunc(method, fixtureKd, testName, logb, func(c *C) {
			c.ResetTimer()
			c.StartTimer()
			defer c.StopTimer()
			c.method.Call([]reflect.Value{reflect.ValueOf(c)})
		})
		return c
	}
	return nil
}

// Run the fixture method with runFixture(), but panic with a fixturePanic{}
// in case the fixture method panics.  This makes it easier to track the
// fixture panic together with other call panics within forkTest().
func (runner *suiteRunner) runFixtureWithPanic(method *methodType, testName string, logb *logger, skipped *bool) *C {
	if skipped != nil && *skipped {
		return nil
	}
	c := runner.runFixture(method, testName, logb)
	if c != nil && c.status != succeededSt {
		if skipped != nil {
			*skipped = c.status == skippedSt
		}
		panic(&fixturePanic{c.status, method})
	}
	return c
}

type fixturePanic struct {
	status funcStatus
	method *methodType
}

// Run the suite test method, together with the test-specific fixture,
// asynchronously.
func (runner *suiteRunner) forkTest(method *methodType) *C {
	testName := method.String()
	return runner.forkCall(method, testKd, testName, nil, func(c *C) {
		var skipped bool
		defer runner.runFixtureWithPanic(runner.tearDownTest, testName, nil, &skipped)
		defer c.StopTimer()
		benchN := 1
		for {
			runner.runFixtureWithPanic(runner.setUpTest, testName, c.logb, &skipped)
			mt := c.method.Type()
			if mt.NumIn() != 1 || mt.In(0) != reflect.TypeOf(c) {
				// Rather than a plain panic, provide a more helpful message when
				// the argument type is incorrect.
				c.status = panickedSt
				c.logArgPanic(c.method, "*check.C")
				return
			}
			if strings.HasPrefix(c.method.Info.Name, "Test") {
				c.ResetTimer()
				c.StartTimer()
				c.method.Call([]reflect.Value{reflect.ValueOf(c)})
				return
			}
			if !strings.HasPrefix(c.method.Info.Name, "Benchmark") {
				panic("unexpected method prefix: " + c.method.Info.Name)
			}

			runtime.GC()
			c.N = benchN
			c.ResetTimer()
			c.StartTimer()
			c.method.Call([]reflect.Value{reflect.ValueOf(c)})
			c.StopTimer()
			if c.status != succeededSt || c.duration >= c.benchTime || benchN >= 1e9 {
				return
			}
			perOpN := int(1e9)
			if c.nsPerOp() != 0 {
				perOpN = int(c.benchTime.Nanoseconds() / c.nsPerOp())
			}

			// Logic taken from the stock testing package:
			// - Run more iterations than we think we'll need for a second (1.5x).
			// - Don't grow too fast in case we had timing errors previously.
			// - Be sure to run at least one more than last time.
			benchN = max(min(perOpN+perOpN/2, 100*benchN), benchN+1)
			benchN = roundUp(benchN)

			skipped = true // Don't run the deferred one if this panics.
			runner.runFixtureWithPanic(runner.tearDownTest, testName, nil, nil)
			skipped = false
		}
	})
}

// Same as forkTest(), but wait for the test to finish before returning.
func (runner *suiteRunner) runTest(method *methodType) *C {
	c := runner.forkTest(method)
	<-c.done
	return c
}

// Helper to mark tests as skipped or missed.  A bit heavy for what
// it does, but it enables homogeneous handling of tracking, including
// nice verbose output.
func (runner *suiteRunner) skipTests(status funcStatus, methods []*methodType) {
	for _, method := range methods {
		runner.runFunc(method, testKd, "", nil, func(c *C) {
			c.status = status
		})
	}
}

// Verify if the fixture arguments are *check.C.  In case of errors,
// log the error as a panic in the fixture method call, and return false.
func (runner *suiteRunner) checkFixtureArgs() bool {
	succeeded := true
	argType := reflect.TypeOf(&C{})
	for _, method := range []*methodType{runner.setUpSuite, runner.tearDownSuite, runner.setUpTest, runner.tearDownTest} {
		if method != nil {
			mt := method.Type()
			if mt.NumIn() != 1 || mt.In(0) != argType {
				succeeded = false
				runner.runFunc(method, fixtureKd, "", nil, func(c *C) {
					c.logArgPanic(method, "*check.C")
					c.status = panickedSt
				})
			}
		}
	}
	return succeeded
}

func (runner *suiteRunner) reportCallStarted(c *C) {
	runner.output.WriteCallStarted("START", c)
}

func (runner *suiteRunner) reportCallDone(c *C) {
	runner.tracker.callDone(c)
	switch c.status {
	case succeededSt:
		if c.mustFail {
			runner.output.WriteCallSuccess("FAIL EXPECTED", c)
		} else {
			runner.output.WriteCallSuccess("PASS", c)
		}
	case skippedSt:
		runner.output.WriteCallSuccess("SKIP", c)
	case failedSt:
		runner.output.WriteCallProblem("FAIL", c)
	case panickedSt:
		runner.output.WriteCallProblem("PANIC", c)
	case fixturePanickedSt:
		// That's a testKd call reporting that its fixture
		// has panicked. The fixture call which caused the
		// panic itself was tracked above. We'll report to
		// aid debugging.
		runner.output.WriteCallProblem("PANIC", c)
	case missedSt:
		runner.output.WriteCallSuccess("MISS", c)
	}
}

// -----------------------------------------------------------------------
// Output writer manages atomic output writing according to settings.

type outputWriter struct {
	m                    sync.Mutex
	writer               io.Writer
	wroteCallProblemLast bool
	Stream               bool
	Verbose              bool
}

func newOutputWriter(writer io.Writer, stream, verbose bool) *outputWriter {
	return &outputWriter{writer: writer, Stream: stream, Verbose: verbose}
}

func (ow *outputWriter) Write(content []byte) (n int, err error) {
	ow.m.Lock()
	n, err = ow.writer.Write(content)
	ow.m.Unlock()
	return
}

func (ow *outputWriter) WriteCallStarted(label string, c *C) {
	if ow.Stream {
		header := renderCallHeader(label, c, "", "\n")
		ow.m.Lock()
		ow.writer.Write([]byte(header))
		ow.m.Unlock()
	}
}

func (ow *outputWriter) WriteCallProblem(label string, c *C) {
	var prefix string
	if !ow.Stream {
		prefix = "\n-----------------------------------" +
			"-----------------------------------\n"
	}
	header := renderCallHeader(label, c, prefix, "\n\n")
	ow.m.Lock()
	ow.wroteCallProblemLast = true
	ow.writer.Write([]byte(header))
	if !ow.Stream {
		c.logb.WriteTo(ow.writer)
	}
	ow.m.Unlock()
}

func (ow *outputWriter) WriteCallSuccess(label string, c *C) {
	if ow.Stream || (ow.Verbose && c.kind == testKd) {
		// TODO Use a buffer here.
		var suffix string
		if c.reason != "" {
			suffix = " (" + c.reason + ")"
		}
		if c.status == succeededSt {
			suffix += "\t" + c.timerString()
		}
		suffix += "\n"
		if ow.Stream {
			suffix += "\n"
		}
		header := renderCallHeader(label, c, "", suffix)
		ow.m.Lock()
		// Resist temptation of using line as prefix above due to race.
		if !ow.Stream && ow.wroteCallProblemLast {
			header = "\n-----------------------------------" +
				"-----------------------------------\n" +
				header
		}
		ow.wroteCallProblemLast = false
		ow.writer.Write([]byte(header))
		ow.m.Unlock()
	}
}

func renderCallHeader(label string, c *C, prefix, suffix string) string {
	pc := c.method.PC()
	return fmt.Sprintf("%s%s: %s: %s%s", prefix, label, niceFuncPath(pc),
		niceFuncName(pc), suffix)
}
