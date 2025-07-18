/*
Ginkgo is a BDD-style testing framework for Golang

The godoc documentation describes Ginkgo's API.  More comprehensive documentation (with examples!) is available at http://onsi.github.io/ginkgo/

Ginkgo's preferred matcher library is [Gomega](http://github.com/onsi/gomega)

Ginkgo on Github: http://github.com/onsi/ginkgo

Ginkgo is MIT-Licensed
*/
package ginkgo

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"reflect"
	"strings"
	"time"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/internal/codelocation"
	"github.com/onsi/ginkgo/internal/global"
	"github.com/onsi/ginkgo/internal/remote"
	"github.com/onsi/ginkgo/internal/testingtproxy"
	"github.com/onsi/ginkgo/internal/writer"
	"github.com/onsi/ginkgo/reporters"
	"github.com/onsi/ginkgo/reporters/stenographer"
	colorable "github.com/onsi/ginkgo/reporters/stenographer/support/go-colorable"
	"github.com/onsi/ginkgo/types"
)

var deprecationTracker = types.NewDeprecationTracker()

const GINKGO_VERSION = config.VERSION
const GINKGO_PANIC = `
Your test failed.
Ginkgo panics to prevent subsequent assertions from running.
Normally Ginkgo rescues this panic so you shouldn't see it.

But, if you make an assertion in a goroutine, Ginkgo can't capture the panic.
To circumvent this, you should call

	defer GinkgoRecover()

at the top of the goroutine that caused this panic.
`

func init() {
	config.Flags(flag.CommandLine, "ginkgo", true)
	GinkgoWriter = writer.New(os.Stdout)
}

//GinkgoWriter implements an io.Writer
//When running in verbose mode any writes to GinkgoWriter will be immediately printed
//to stdout.  Otherwise, GinkgoWriter will buffer any writes produced during the current test and flush them to screen
//only if the current test fails.
var GinkgoWriter io.Writer

//The interface by which Ginkgo receives *testing.T
type GinkgoTestingT interface {
	Fail()
}

//GinkgoRandomSeed returns the seed used to randomize spec execution order.  It is
//useful for seeding your own pseudorandom number generators (PRNGs) to ensure
//consistent executions from run to run, where your tests contain variability (for
//example, when selecting random test data).
func GinkgoRandomSeed() int64 {
	return config.GinkgoConfig.RandomSeed
}

//GinkgoParallelNode is deprecated, use GinkgoParallelProcess instead
func GinkgoParallelNode() int {
	deprecationTracker.TrackDeprecation(types.Deprecations.ParallelNode(), codelocation.New(1))
	return GinkgoParallelProcess()
}

//GinkgoParallelProcess returns the parallel process number for the current ginkgo process
//The process number is 1-indexed
func GinkgoParallelProcess() int {
	return config.GinkgoConfig.ParallelNode
}

//Some matcher libraries or legacy codebases require a *testing.T
//GinkgoT implements an interface analogous to *testing.T and can be used if
//the library in question accepts *testing.T through an interface
//
// For example, with testify:
// assert.Equal(GinkgoT(), 123, 123, "they should be equal")
//
// Or with gomock:
// gomock.NewController(GinkgoT())
//
// GinkgoT() takes an optional offset argument that can be used to get the
// correct line number associated with the failure.
func GinkgoT(optionalOffset ...int) GinkgoTInterface {
	offset := 3
	if len(optionalOffset) > 0 {
		offset = optionalOffset[0]
	}
	failedFunc := func() bool {
		return CurrentGinkgoTestDescription().Failed
	}
	nameFunc := func() string {
		return CurrentGinkgoTestDescription().FullTestText
	}
	return testingtproxy.New(GinkgoWriter, Fail, Skip, failedFunc, nameFunc, offset)
}

//The interface returned by GinkgoT().  This covers most of the methods
//in the testing package's T.
type GinkgoTInterface interface {
	Cleanup(func())
	Setenv(key, value string)
	Error(args ...interface{})
	Errorf(format string, args ...interface{})
	Fail()
	FailNow()
	Failed() bool
	Fatal(args ...interface{})
	Fatalf(format string, args ...interface{})
	Helper()
	Log(args ...interface{})
	Logf(format string, args ...interface{})
	Name() string
	Parallel()
	Skip(args ...interface{})
	SkipNow()
	Skipf(format string, args ...interface{})
	Skipped() bool
	TempDir() string
}

//Custom Ginkgo test reporters must implement the Reporter interface.
//
//The custom reporter is passed in a SuiteSummary when the suite begins and ends,
//and a SpecSummary just before a spec begins and just after a spec ends
type Reporter reporters.Reporter

//Asynchronous specs are given a channel of the Done type.  You must close or write to the channel
//to tell Ginkgo that your async test is done.
type Done chan<- interface{}

//GinkgoTestDescription represents the information about the current running test returned by CurrentGinkgoTestDescription
//	FullTestText: a concatenation of ComponentTexts and the TestText
//	ComponentTexts: a list of all texts for the Describes & Contexts leading up to the current test
//	TestText: the text in the actual It or Measure node
//	IsMeasurement: true if the current test is a measurement
//	FileName: the name of the file containing the current test
//	LineNumber: the line number for the current test
//	Failed: if the current test has failed, this will be true (useful in an AfterEach)
type GinkgoTestDescription struct {
	FullTestText   string
	ComponentTexts []string
	TestText       string

	IsMeasurement bool

	FileName   string
	LineNumber int

	Failed   bool
	Duration time.Duration
}

//CurrentGinkgoTestDescripton returns information about the current running test.
func CurrentGinkgoTestDescription() GinkgoTestDescription {
	summary, ok := global.Suite.CurrentRunningSpecSummary()
	if !ok {
		return GinkgoTestDescription{}
	}

	subjectCodeLocation := summary.ComponentCodeLocations[len(summary.ComponentCodeLocations)-1]

	return GinkgoTestDescription{
		ComponentTexts: summary.ComponentTexts[1:],
		FullTestText:   strings.Join(summary.ComponentTexts[1:], " "),
		TestText:       summary.ComponentTexts[len(summary.ComponentTexts)-1],
		IsMeasurement:  summary.IsMeasurement,
		FileName:       subjectCodeLocation.FileName,
		LineNumber:     subjectCodeLocation.LineNumber,
		Failed:         summary.HasFailureState(),
		Duration:       summary.RunTime,
	}
}

//Measurement tests receive a Benchmarker.
//
//You use the Time() function to time how long the passed in body function takes to run
//You use the RecordValue() function to track arbitrary numerical measurements.
//The RecordValueWithPrecision() function can be used alternatively to provide the unit
//and resolution of the numeric measurement.
//The optional info argument is passed to the test reporter and can be used to
// provide the measurement data to a custom reporter with context.
//
//See http://onsi.github.io/ginkgo/#benchmark_tests for more details
type Benchmarker interface {
	Time(name string, body func(), info ...interface{}) (elapsedTime time.Duration)
	RecordValue(name string, value float64, info ...interface{})
	RecordValueWithPrecision(name string, value float64, units string, precision int, info ...interface{})
}

//RunSpecs is the entry point for the Ginkgo test runner.
//You must call this within a Golang testing TestX(t *testing.T) function.
//
//To bootstrap a test suite you can use the Ginkgo CLI:
//
//	ginkgo bootstrap
func RunSpecs(t GinkgoTestingT, description string) bool {
	specReporters := []Reporter{buildDefaultReporter()}
	if config.DefaultReporterConfig.ReportFile != "" {
		reportFile := config.DefaultReporterConfig.ReportFile
		specReporters[0] = reporters.NewJUnitReporter(reportFile)
		specReporters = append(specReporters, buildDefaultReporter())
	}
	return runSpecsWithCustomReporters(t, description, specReporters)
}

//To run your tests with Ginkgo's default reporter and your custom reporter(s), replace
//RunSpecs() with this method.
func RunSpecsWithDefaultAndCustomReporters(t GinkgoTestingT, description string, specReporters []Reporter) bool {
	deprecationTracker.TrackDeprecation(types.Deprecations.CustomReporter())
	specReporters = append(specReporters, buildDefaultReporter())
	return runSpecsWithCustomReporters(t, description, specReporters)
}

//To run your tests with your custom reporter(s) (and *not* Ginkgo's default reporter), replace
//RunSpecs() with this method.  Note that parallel tests will not work correctly without the default reporter
func RunSpecsWithCustomReporters(t GinkgoTestingT, description string, specReporters []Reporter) bool {
	deprecationTracker.TrackDeprecation(types.Deprecations.CustomReporter())
	return runSpecsWithCustomReporters(t, description, specReporters)
}

func runSpecsWithCustomReporters(t GinkgoTestingT, description string, specReporters []Reporter) bool {
	writer := GinkgoWriter.(*writer.Writer)
	writer.SetStream(config.DefaultReporterConfig.Verbose)
	reporters := make([]reporters.Reporter, len(specReporters))
	for i, reporter := range specReporters {
		reporters[i] = reporter
	}
	passed, hasFocusedTests := global.Suite.Run(t, description, reporters, writer, config.GinkgoConfig)

	if deprecationTracker.DidTrackDeprecations() {
		fmt.Fprintln(colorable.NewColorableStderr(), deprecationTracker.DeprecationsReport())
	}

	if passed && hasFocusedTests && strings.TrimSpace(os.Getenv("GINKGO_EDITOR_INTEGRATION")) == "" {
		fmt.Println("PASS | FOCUSED")
		os.Exit(types.GINKGO_FOCUS_EXIT_CODE)
	}
	return passed
}

func buildDefaultReporter() Reporter {
	remoteReportingServer := config.GinkgoConfig.StreamHost
	if remoteReportingServer == "" {
		stenographer := stenographer.New(!config.DefaultReporterConfig.NoColor, config.GinkgoConfig.FlakeAttempts > 1, colorable.NewColorableStdout())
		return reporters.NewDefaultReporter(config.DefaultReporterConfig, stenographer)
	} else {
		debugFile := ""
		if config.GinkgoConfig.DebugParallel {
			debugFile = fmt.Sprintf("ginkgo-node-%d.log", config.GinkgoConfig.ParallelNode)
		}
		return remote.NewForwardingReporter(config.DefaultReporterConfig, remoteReportingServer, &http.Client{}, remote.NewOutputInterceptor(), GinkgoWriter.(*writer.Writer), debugFile)
	}
}

//Skip notifies Ginkgo that the current spec was skipped.
func Skip(message string, callerSkip ...int) {
	skip := 0
	if len(callerSkip) > 0 {
		skip = callerSkip[0]
	}

	global.Failer.Skip(message, codelocation.New(skip+1))
	panic(GINKGO_PANIC)
}

//Fail notifies Ginkgo that the current spec has failed. (Gomega will call Fail for you automatically when an assertion fails.)
func Fail(message string, callerSkip ...int) {
	skip := 0
	if len(callerSkip) > 0 {
		skip = callerSkip[0]
	}

	global.Failer.Fail(message, codelocation.New(skip+1))
	panic(GINKGO_PANIC)
}

//GinkgoRecover should be deferred at the top of any spawned goroutine that (may) call `Fail`
//Since Gomega assertions call fail, you should throw a `defer GinkgoRecover()` at the top of any goroutine that
//calls out to Gomega
//
//Here's why: Ginkgo's `Fail` method records the failure and then panics to prevent
//further assertions from running.  This panic must be recovered.  Ginkgo does this for you
//if the panic originates in a Ginkgo node (an It, BeforeEach, etc...)
//
//Unfortunately, if a panic originates on a goroutine *launched* from one of these nodes there's no
//way for Ginkgo to rescue the panic.  To do this, you must remember to `defer GinkgoRecover()` at the top of such a goroutine.
func GinkgoRecover() {
	e := recover()
	if e != nil {
		global.Failer.Panic(codelocation.New(1), e)
	}
}

//Describe blocks allow you to organize your specs.  A Describe block can contain any number of
//BeforeEach, AfterEach, JustBeforeEach, It, and Measurement blocks.
//
//In addition you can nest Describe, Context and When blocks.  Describe, Context and When blocks are functionally
//equivalent.  The difference is purely semantic -- you typically Describe the behavior of an object
//or method and, within that Describe, outline a number of Contexts and Whens.
func Describe(text string, body func()) bool {
	global.Suite.PushContainerNode(text, body, types.FlagTypeNone, codelocation.New(1))
	return true
}

//You can focus the tests within a describe block using FDescribe
func FDescribe(text string, body func()) bool {
	global.Suite.PushContainerNode(text, body, types.FlagTypeFocused, codelocation.New(1))
	return true
}

//You can mark the tests within a describe block as pending using PDescribe
func PDescribe(text string, body func()) bool {
	global.Suite.PushContainerNode(text, body, types.FlagTypePending, codelocation.New(1))
	return true
}

//You can mark the tests within a describe block as pending using XDescribe
func XDescribe(text string, body func()) bool {
	global.Suite.PushContainerNode(text, body, types.FlagTypePending, codelocation.New(1))
	return true
}

//Context blocks allow you to organize your specs.  A Context block can contain any number of
//BeforeEach, AfterEach, JustBeforeEach, It, and Measurement blocks.
//
//In addition you can nest Describe, Context and When blocks.  Describe, Context and When blocks are functionally
//equivalent.  The difference is purely semantic -- you typical Describe the behavior of an object
//or method and, within that Describe, outline a number of Contexts and Whens.
func Context(text string, body func()) bool {
	global.Suite.PushContainerNode(text, body, types.FlagTypeNone, codelocation.New(1))
	return true
}

//You can focus the tests within a describe block using FContext
func FContext(text string, body func()) bool {
	global.Suite.PushContainerNode(text, body, types.FlagTypeFocused, codelocation.New(1))
	return true
}

//You can mark the tests within a describe block as pending using PContext
func PContext(text string, body func()) bool {
	global.Suite.PushContainerNode(text, body, types.FlagTypePending, codelocation.New(1))
	return true
}

//You can mark the tests within a describe block as pending using XContext
func XContext(text string, body func()) bool {
	global.Suite.PushContainerNode(text, body, types.FlagTypePending, codelocation.New(1))
	return true
}

//When blocks allow you to organize your specs.  A When block can contain any number of
//BeforeEach, AfterEach, JustBeforeEach, It, and Measurement blocks.
//
//In addition you can nest Describe, Context and When blocks.  Describe, Context and When blocks are functionally
//equivalent.  The difference is purely semantic -- you typical Describe the behavior of an object
//or method and, within that Describe, outline a number of Contexts and Whens.
func When(text string, body func()) bool {
	global.Suite.PushContainerNode("when "+text, body, types.FlagTypeNone, codelocation.New(1))
	return true
}

//You can focus the tests within a describe block using FWhen
func FWhen(text string, body func()) bool {
	global.Suite.PushContainerNode("when "+text, body, types.FlagTypeFocused, codelocation.New(1))
	return true
}

//You can mark the tests within a describe block as pending using PWhen
func PWhen(text string, body func()) bool {
	global.Suite.PushContainerNode("when "+text, body, types.FlagTypePending, codelocation.New(1))
	return true
}

//You can mark the tests within a describe block as pending using XWhen
func XWhen(text string, body func()) bool {
	global.Suite.PushContainerNode("when "+text, body, types.FlagTypePending, codelocation.New(1))
	return true
}

//It blocks contain your test code and assertions.  You cannot nest any other Ginkgo blocks
//within an It block.
//
//Ginkgo will normally run It blocks synchronously.  To perform asynchronous tests, pass a
//function that accepts a Done channel.  When you do this, you can also provide an optional timeout.
func It(text string, body interface{}, timeout ...float64) bool {
	validateBodyFunc(body, codelocation.New(1))
	global.Suite.PushItNode(text, body, types.FlagTypeNone, codelocation.New(1), parseTimeout(timeout...))
	return true
}

//You can focus individual Its using FIt
func FIt(text string, body interface{}, timeout ...float64) bool {
	validateBodyFunc(body, codelocation.New(1))
	global.Suite.PushItNode(text, body, types.FlagTypeFocused, codelocation.New(1), parseTimeout(timeout...))
	return true
}

//You can mark Its as pending using PIt
func PIt(text string, _ ...interface{}) bool {
	global.Suite.PushItNode(text, func() {}, types.FlagTypePending, codelocation.New(1), 0)
	return true
}

//You can mark Its as pending using XIt
func XIt(text string, _ ...interface{}) bool {
	global.Suite.PushItNode(text, func() {}, types.FlagTypePending, codelocation.New(1), 0)
	return true
}

//Specify blocks are aliases for It blocks and allow for more natural wording in situations
//which "It" does not fit into a natural sentence flow. All the same protocols apply for Specify blocks
//which apply to It blocks.
func Specify(text string, body interface{}, timeout ...float64) bool {
	validateBodyFunc(body, codelocation.New(1))
	global.Suite.PushItNode(text, body, types.FlagTypeNone, codelocation.New(1), parseTimeout(timeout...))
	return true
}

//You can focus individual Specifys using FSpecify
func FSpecify(text string, body interface{}, timeout ...float64) bool {
	validateBodyFunc(body, codelocation.New(1))
	global.Suite.PushItNode(text, body, types.FlagTypeFocused, codelocation.New(1), parseTimeout(timeout...))
	return true
}

//You can mark Specifys as pending using PSpecify
func PSpecify(text string, is ...interface{}) bool {
	global.Suite.PushItNode(text, func() {}, types.FlagTypePending, codelocation.New(1), 0)
	return true
}

//You can mark Specifys as pending using XSpecify
func XSpecify(text string, is ...interface{}) bool {
	global.Suite.PushItNode(text, func() {}, types.FlagTypePending, codelocation.New(1), 0)
	return true
}

//By allows you to better document large Its.
//
//Generally you should try to keep your Its short and to the point.  This is not always possible, however,
//especially in the context of integration tests that capture a particular workflow.
//
//By allows you to document such flows.  By must be called within a runnable node (It, BeforeEach, Measure, etc...)
//By will simply log the passed in text to the GinkgoWriter.  If By is handed a function it will immediately run the function.
func By(text string, callbacks ...func()) {
	preamble := "\x1b[1mSTEP\x1b[0m"
	if config.DefaultReporterConfig.NoColor {
		preamble = "STEP"
	}
	fmt.Fprintln(GinkgoWriter, preamble+": "+text)
	if len(callbacks) == 1 {
		callbacks[0]()
	}
	if len(callbacks) > 1 {
		panic("just one callback per By, please")
	}
}

//Measure blocks run the passed in body function repeatedly (determined by the samples argument)
//and accumulate metrics provided to the Benchmarker by the body function.
//
//The body function must have the signature:
//	func(b Benchmarker)
func Measure(text string, body interface{}, samples int) bool {
	deprecationTracker.TrackDeprecation(types.Deprecations.Measure(), codelocation.New(1))
	global.Suite.PushMeasureNode(text, body, types.FlagTypeNone, codelocation.New(1), samples)
	return true
}

//You can focus individual Measures using FMeasure
func FMeasure(text string, body interface{}, samples int) bool {
	deprecationTracker.TrackDeprecation(types.Deprecations.Measure(), codelocation.New(1))
	global.Suite.PushMeasureNode(text, body, types.FlagTypeFocused, codelocation.New(1), samples)
	return true
}

//You can mark Measurements as pending using PMeasure
func PMeasure(text string, _ ...interface{}) bool {
	deprecationTracker.TrackDeprecation(types.Deprecations.Measure(), codelocation.New(1))
	global.Suite.PushMeasureNode(text, func(b Benchmarker) {}, types.FlagTypePending, codelocation.New(1), 0)
	return true
}

//You can mark Measurements as pending using XMeasure
func XMeasure(text string, _ ...interface{}) bool {
	deprecationTracker.TrackDeprecation(types.Deprecations.Measure(), codelocation.New(1))
	global.Suite.PushMeasureNode(text, func(b Benchmarker) {}, types.FlagTypePending, codelocation.New(1), 0)
	return true
}

//BeforeSuite blocks are run just once before any specs are run.  When running in parallel, each
//parallel node process will call BeforeSuite.
//
//BeforeSuite blocks can be made asynchronous by providing a body function that accepts a Done channel
//
//You may only register *one* BeforeSuite handler per test suite.  You typically do so in your bootstrap file at the top level.
func BeforeSuite(body interface{}, timeout ...float64) bool {
	validateBodyFunc(body, codelocation.New(1))
	global.Suite.SetBeforeSuiteNode(body, codelocation.New(1), parseTimeout(timeout...))
	return true
}

//AfterSuite blocks are *always* run after all the specs regardless of whether specs have passed or failed.
//Moreover, if Ginkgo receives an interrupt signal (^C) it will attempt to run the AfterSuite before exiting.
//
//When running in parallel, each parallel node process will call AfterSuite.
//
//AfterSuite blocks can be made asynchronous by providing a body function that accepts a Done channel
//
//You may only register *one* AfterSuite handler per test suite.  You typically do so in your bootstrap file at the top level.
func AfterSuite(body interface{}, timeout ...float64) bool {
	validateBodyFunc(body, codelocation.New(1))
	global.Suite.SetAfterSuiteNode(body, codelocation.New(1), parseTimeout(timeout...))
	return true
}

//SynchronizedBeforeSuite blocks are primarily meant to solve the problem of setting up singleton external resources shared across
//nodes when running tests in parallel.  For example, say you have a shared database that you can only start one instance of that
//must be used in your tests.  When running in parallel, only one node should set up the database and all other nodes should wait
//until that node is done before running.
//
//SynchronizedBeforeSuite accomplishes this by taking *two* function arguments.  The first is only run on parallel node #1.  The second is
//run on all nodes, but *only* after the first function completes successfully.  Ginkgo also makes it possible to send data from the first function (on Node 1)
//to the second function (on all the other nodes).
//
//The functions have the following signatures.  The first function (which only runs on node 1) has the signature:
//
//	func() []byte
//
//or, to run asynchronously:
//
//	func(done Done) []byte
//
//The byte array returned by the first function is then passed to the second function, which has the signature:
//
//	func(data []byte)
//
//or, to run asynchronously:
//
//	func(data []byte, done Done)
//
//Here's a simple pseudo-code example that starts a shared database on Node 1 and shares the database's address with the other nodes:
//
//	var dbClient db.Client
//	var dbRunner db.Runner
//
//	var _ = SynchronizedBeforeSuite(func() []byte {
//		dbRunner = db.NewRunner()
//		err := dbRunner.Start()
//		Ω(err).ShouldNot(HaveOccurred())
//		return []byte(dbRunner.URL)
//	}, func(data []byte) {
//		dbClient = db.NewClient()
//		err := dbClient.Connect(string(data))
//		Ω(err).ShouldNot(HaveOccurred())
//	})
func SynchronizedBeforeSuite(node1Body interface{}, allNodesBody interface{}, timeout ...float64) bool {
	global.Suite.SetSynchronizedBeforeSuiteNode(
		node1Body,
		allNodesBody,
		codelocation.New(1),
		parseTimeout(timeout...),
	)
	return true
}

//SynchronizedAfterSuite blocks complement the SynchronizedBeforeSuite blocks in solving the problem of setting up
//external singleton resources shared across nodes when running tests in parallel.
//
//SynchronizedAfterSuite accomplishes this by taking *two* function arguments.  The first runs on all nodes.  The second runs only on parallel node #1
//and *only* after all other nodes have finished and exited.  This ensures that node 1, and any resources it is running, remain alive until
//all other nodes are finished.
//
//Both functions have the same signature: either func() or func(done Done) to run asynchronously.
//
//Here's a pseudo-code example that complements that given in SynchronizedBeforeSuite.  Here, SynchronizedAfterSuite is used to tear down the shared database
//only after all nodes have finished:
//
//	var _ = SynchronizedAfterSuite(func() {
//		dbClient.Cleanup()
//	}, func() {
//		dbRunner.Stop()
//	})
func SynchronizedAfterSuite(allNodesBody interface{}, node1Body interface{}, timeout ...float64) bool {
	global.Suite.SetSynchronizedAfterSuiteNode(
		allNodesBody,
		node1Body,
		codelocation.New(1),
		parseTimeout(timeout...),
	)
	return true
}

//BeforeEach blocks are run before It blocks.  When multiple BeforeEach blocks are defined in nested
//Describe and Context blocks the outermost BeforeEach blocks are run first.
//
//Like It blocks, BeforeEach blocks can be made asynchronous by providing a body function that accepts
//a Done channel
func BeforeEach(body interface{}, timeout ...float64) bool {
	validateBodyFunc(body, codelocation.New(1))
	global.Suite.PushBeforeEachNode(body, codelocation.New(1), parseTimeout(timeout...))
	return true
}

//JustBeforeEach blocks are run before It blocks but *after* all BeforeEach blocks.  For more details,
//read the [documentation](http://onsi.github.io/ginkgo/#separating_creation_and_configuration_)
//
//Like It blocks, BeforeEach blocks can be made asynchronous by providing a body function that accepts
//a Done channel
func JustBeforeEach(body interface{}, timeout ...float64) bool {
	validateBodyFunc(body, codelocation.New(1))
	global.Suite.PushJustBeforeEachNode(body, codelocation.New(1), parseTimeout(timeout...))
	return true
}

//JustAfterEach blocks are run after It blocks but *before* all AfterEach blocks.  For more details,
//read the [documentation](http://onsi.github.io/ginkgo/#separating_creation_and_configuration_)
//
//Like It blocks, JustAfterEach blocks can be made asynchronous by providing a body function that accepts
//a Done channel
func JustAfterEach(body interface{}, timeout ...float64) bool {
	validateBodyFunc(body, codelocation.New(1))
	global.Suite.PushJustAfterEachNode(body, codelocation.New(1), parseTimeout(timeout...))
	return true
}

//AfterEach blocks are run after It blocks.   When multiple AfterEach blocks are defined in nested
//Describe and Context blocks the innermost AfterEach blocks are run first.
//
//Like It blocks, AfterEach blocks can be made asynchronous by providing a body function that accepts
//a Done channel
func AfterEach(body interface{}, timeout ...float64) bool {
	validateBodyFunc(body, codelocation.New(1))
	global.Suite.PushAfterEachNode(body, codelocation.New(1), parseTimeout(timeout...))
	return true
}

func validateBodyFunc(body interface{}, cl types.CodeLocation) {
	t := reflect.TypeOf(body)
	if t.Kind() != reflect.Func {
		return
	}

	if t.NumOut() > 0 {
		return
	}

	if t.NumIn() == 0 {
		return
	}

	if t.In(0) == reflect.TypeOf(make(Done)) {
		deprecationTracker.TrackDeprecation(types.Deprecations.Async(), cl)
	}
}

func parseTimeout(timeout ...float64) time.Duration {
	if len(timeout) == 0 {
		return global.DefaultTimeout
	} else {
		return time.Duration(timeout[0] * float64(time.Second))
	}
}
