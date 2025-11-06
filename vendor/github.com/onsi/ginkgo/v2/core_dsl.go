/*
Ginkgo is a testing framework for Go designed to help you write expressive tests.
https://github.com/onsi/ginkgo
MIT-Licensed

The godoc documentation outlines Ginkgo's API.  Since Ginkgo is a Domain-Specific Language it is important to
build a mental model for Ginkgo - the narrative documentation at https://onsi.github.io/ginkgo/ is designed to help you do that.
You should start there - even a brief skim will be helpful.  At minimum you should skim through the https://onsi.github.io/ginkgo/#getting-started chapter.

Ginkgo's is best paired with the Gomega matcher library: https://github.com/onsi/gomega

You can run Ginkgo specs with go test - however we recommend using the ginkgo cli.  It enables functionality
that go test does not (especially running suites in parallel).  You can learn more at https://onsi.github.io/ginkgo/#ginkgo-cli-overview
or by running 'ginkgo help'.
*/
package ginkgo

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-logr/logr"
	"github.com/onsi/ginkgo/v2/formatter"
	"github.com/onsi/ginkgo/v2/internal"
	"github.com/onsi/ginkgo/v2/internal/global"
	"github.com/onsi/ginkgo/v2/internal/interrupt_handler"
	"github.com/onsi/ginkgo/v2/internal/parallel_support"
	"github.com/onsi/ginkgo/v2/reporters"
	"github.com/onsi/ginkgo/v2/types"
)

const GINKGO_VERSION = types.VERSION

var flagSet types.GinkgoFlagSet
var deprecationTracker = types.NewDeprecationTracker()
var suiteConfig = types.NewDefaultSuiteConfig()
var reporterConfig = types.NewDefaultReporterConfig()
var suiteDidRun = false
var outputInterceptor internal.OutputInterceptor
var client parallel_support.Client

func init() {
	var err error
	flagSet, err = types.BuildTestSuiteFlagSet(&suiteConfig, &reporterConfig)
	exitIfErr(err)
	writer := internal.NewWriter(os.Stdout)
	GinkgoWriter = writer
	GinkgoLogr = internal.GinkgoLogrFunc(writer)
}

func exitIfErr(err error) {
	if err != nil {
		if outputInterceptor != nil {
			outputInterceptor.Shutdown()
		}
		if client != nil {
			client.Close()
		}
		fmt.Fprintln(formatter.ColorableStdErr, err.Error())
		os.Exit(1)
	}
}

func exitIfErrors(errors []error) {
	if len(errors) > 0 {
		if outputInterceptor != nil {
			outputInterceptor.Shutdown()
		}
		if client != nil {
			client.Close()
		}
		for _, err := range errors {
			fmt.Fprintln(formatter.ColorableStdErr, err.Error())
		}
		os.Exit(1)
	}
}

// The interface implemented by GinkgoWriter
type GinkgoWriterInterface interface {
	io.Writer

	Print(a ...any)
	Printf(format string, a ...any)
	Println(a ...any)

	TeeTo(writer io.Writer)
	ClearTeeWriters()
}

/*
SpecContext is the context object passed into nodes that are subject to a timeout or need to be notified of an interrupt.  It implements the standard context.Context interface but also contains additional helpers to provide an extensibility point for Ginkgo.  (As an example, Gomega's Eventually can use the methods defined on SpecContext to provide deeper integration with Ginkgo).

You can do anything with SpecContext that you do with a typical context.Context including wrapping it with any of the context.With* methods.

Ginkgo will cancel the SpecContext when a node is interrupted (e.g. by the user sending an interrupt signal) or when a node has exceeded its allowed run-time.  Note, however, that even in cases where a node has a deadline, SpecContext will not return a deadline via .Deadline().  This is because Ginkgo does not use a WithDeadline() context to model node deadlines as Ginkgo needs control over the precise timing of the context cancellation to ensure it can provide an accurate progress report at the moment of cancellation.
*/
type SpecContext = internal.SpecContext

/*
GinkgoWriter implements a GinkgoWriterInterface and io.Writer

When running in verbose mode (ginkgo -v) any writes to GinkgoWriter will be immediately printed
to stdout.  Otherwise, GinkgoWriter will buffer any writes produced during the current test and flush them to screen
only if the current test fails.

GinkgoWriter also provides convenience Print, Printf and Println methods and allows you to tee to a custom writer via GinkgoWriter.TeeTo(writer).
Writes to GinkgoWriter are immediately sent to any registered TeeTo() writers.  You can unregister all TeeTo() Writers with GinkgoWriter.ClearTeeWriters()

You can learn more at https://onsi.github.io/ginkgo/#logging-output
*/
var GinkgoWriter GinkgoWriterInterface

/*
GinkgoLogr is a logr.Logger that writes to GinkgoWriter
*/
var GinkgoLogr logr.Logger

// The interface by which Ginkgo receives *testing.T
type GinkgoTestingT interface {
	Fail()
}

/*
GinkgoConfiguration returns the configuration of the current suite.

The first return value is the SuiteConfig which controls aspects of how the suite runs,
the second return value is the ReporterConfig which controls aspects of how Ginkgo's default
reporter emits output.

Mutating the returned configurations has no effect.  To reconfigure Ginkgo programmatically you need
to pass in your mutated copies into RunSpecs().

You can learn more at https://onsi.github.io/ginkgo/#overriding-ginkgos-command-line-configuration-in-the-suite
*/
func GinkgoConfiguration() (types.SuiteConfig, types.ReporterConfig) {
	return suiteConfig, reporterConfig
}

/*
GinkgoRandomSeed returns the seed used to randomize spec execution order.  It is
useful for seeding your own pseudorandom number generators to ensure
consistent executions from run to run, where your tests contain variability (for
example, when selecting random spec data).

You can learn more at https://onsi.github.io/ginkgo/#spec-randomization
*/
func GinkgoRandomSeed() int64 {
	return suiteConfig.RandomSeed
}

/*
GinkgoParallelProcess returns the parallel process number for the current ginkgo process
The process number is 1-indexed.  You can use GinkgoParallelProcess() to shard access to shared
resources across your suites.  You can learn more about patterns for sharding at https://onsi.github.io/ginkgo/#patterns-for-parallel-integration-specs

For more on how specs are parallelized in Ginkgo, see http://onsi.github.io/ginkgo/#spec-parallelization
*/
func GinkgoParallelProcess() int {
	return suiteConfig.ParallelProcess
}

/*
GinkgoHelper marks the function it's called in as a test helper.  When a failure occurs inside a helper function, Ginkgo will skip the helper when analyzing the stack trace to identify where the failure occurred.

This is an alternative, simpler, mechanism to passing in a skip offset when calling Fail or using Gomega.
*/
func GinkgoHelper() {
	types.MarkAsHelper(1)
}

/*
GinkgoLabelFilter() returns the label filter configured for this suite via `--label-filter`.

You can use this to manually check if a set of labels would satisfy the filter via:

	if (Label("cat", "dog").MatchesLabelFilter(GinkgoLabelFilter())) {
		//...
	}
*/
func GinkgoLabelFilter() string {
	suiteConfig, _ := GinkgoConfiguration()
	return suiteConfig.LabelFilter
}

/*
GinkgoSemVerFilter() returns the semantic version filter configured for this suite via `--sem-ver-filter`.

You can use this to manually check if a set of semantic version constraints would satisfy the filter via:

	if (SemVerConstraint("> 2.6.0", "< 2.8.0").MatchesSemVerFilter(GinkgoSemVerFilter())) {
		//...
	}
*/
func GinkgoSemVerFilter() string {
	suiteConfig, _ := GinkgoConfiguration()
	return suiteConfig.SemVerFilter
}

/*
PauseOutputInterception() pauses Ginkgo's output interception.  This is only relevant
when running in parallel and output to stdout/stderr is being intercepted.  You generally
don't need to call this function - however there are cases when Ginkgo's output interception
mechanisms can interfere with external processes launched by the test process.

In particular, if an external process is launched that has cmd.Stdout/cmd.Stderr set to os.Stdout/os.Stderr
then Ginkgo's output interceptor will hang.  To circumvent this, set cmd.Stdout/cmd.Stderr to GinkgoWriter.
If, for some reason, you aren't able to do that, you can PauseOutputInterception() before starting the process
then ResumeOutputInterception() after starting it.

Note that PauseOutputInterception() does not cause stdout writes to print to the console -
this simply stops intercepting and storing stdout writes to an internal buffer.
*/
func PauseOutputInterception() {
	if outputInterceptor == nil {
		return
	}
	outputInterceptor.PauseIntercepting()
}

// ResumeOutputInterception() - see docs for PauseOutputInterception()
func ResumeOutputInterception() {
	if outputInterceptor == nil {
		return
	}
	outputInterceptor.ResumeIntercepting()
}

/*
RunSpecs is the entry point for the Ginkgo spec runner.

You must call this within a Golang testing TestX(t *testing.T) function.
If you bootstrapped your suite with "ginkgo bootstrap" this is already
done for you.

Ginkgo is typically configured via command-line flags.  This configuration
can be overridden, however, and passed into RunSpecs as optional arguments:

	func TestMySuite(t *testing.T)  {
		RegisterFailHandler(gomega.Fail)
		// fetch the current config
		suiteConfig, reporterConfig := GinkgoConfiguration()
		// adjust it
		suiteConfig.SkipStrings = []string{"NEVER-RUN"}
		reporterConfig.FullTrace = true
		// pass it in to RunSpecs
		RunSpecs(t, "My Suite", suiteConfig, reporterConfig)
	}

Note that some configuration changes can lead to undefined behavior.  For example,
you should not change ParallelProcess or ParallelTotal as the Ginkgo CLI is responsible
for setting these and orchestrating parallel specs across the parallel processes.  See http://onsi.github.io/ginkgo/#spec-parallelization
for more on how specs are parallelized in Ginkgo.

You can also pass suite-level Label() decorators to RunSpecs.  The passed-in labels will apply to all specs in the suite.
*/
func RunSpecs(t GinkgoTestingT, description string, args ...any) bool {
	if suiteDidRun {
		exitIfErr(types.GinkgoErrors.RerunningSuite())
	}
	suiteDidRun = true
	err := global.PushClone()
	if err != nil {
		exitIfErr(err)
	}
	defer global.PopClone()

	suiteLabels, suiteSemVerConstraints, suiteAroundNodes := extractSuiteConfiguration(args)

	var reporter reporters.Reporter
	if suiteConfig.ParallelTotal == 1 {
		reporter = reporters.NewDefaultReporter(reporterConfig, formatter.ColorableStdOut)
		outputInterceptor = internal.NoopOutputInterceptor{}
		client = nil
	} else {
		reporter = reporters.NoopReporter{}
		switch strings.ToLower(suiteConfig.OutputInterceptorMode) {
		case "swap":
			outputInterceptor = internal.NewOSGlobalReassigningOutputInterceptor()
		case "none":
			outputInterceptor = internal.NoopOutputInterceptor{}
		default:
			outputInterceptor = internal.NewOutputInterceptor()
		}
		client = parallel_support.NewClient(suiteConfig.ParallelHost)
		if !client.Connect() {
			client = nil
			exitIfErr(types.GinkgoErrors.UnreachableParallelHost(suiteConfig.ParallelHost))
		}
		defer client.Close()
	}

	writer := GinkgoWriter.(*internal.Writer)
	if reporterConfig.Verbosity().GTE(types.VerbosityLevelVerbose) && suiteConfig.ParallelTotal == 1 {
		writer.SetMode(internal.WriterModeStreamAndBuffer)
	} else {
		writer.SetMode(internal.WriterModeBufferOnly)
	}

	if reporterConfig.WillGenerateReport() {
		registerReportAfterSuiteNodeForAutogeneratedReports(reporterConfig)
	}

	err = global.Suite.BuildTree()
	exitIfErr(err)
	suitePath, err := getwd()
	exitIfErr(err)
	suitePath, err = filepath.Abs(suitePath)
	exitIfErr(err)

	passed, hasFocusedTests := global.Suite.Run(description, suiteLabels, suiteSemVerConstraints, suiteAroundNodes, suitePath, global.Failer, reporter, writer, outputInterceptor, interrupt_handler.NewInterruptHandler(client), client, internal.RegisterForProgressSignal, suiteConfig)
	outputInterceptor.Shutdown()

	flagSet.ValidateDeprecations(deprecationTracker)
	if deprecationTracker.DidTrackDeprecations() {
		fmt.Fprintln(formatter.ColorableStdErr, deprecationTracker.DeprecationsReport())
	}

	if !passed {
		t.Fail()
	}

	if passed && hasFocusedTests && strings.TrimSpace(os.Getenv("GINKGO_EDITOR_INTEGRATION")) == "" {
		fmt.Println("PASS | FOCUSED")
		os.Exit(types.GINKGO_FOCUS_EXIT_CODE)
	}
	return passed
}

func extractSuiteConfiguration(args []any) (Labels, SemVerConstraints, types.AroundNodes) {
	suiteLabels := Labels{}
	suiteSemVerConstraints := SemVerConstraints{}
	aroundNodes := types.AroundNodes{}
	configErrors := []error{}
	for _, arg := range args {
		switch arg := arg.(type) {
		case types.SuiteConfig:
			suiteConfig = arg
		case types.ReporterConfig:
			reporterConfig = arg
		case Labels:
			suiteLabels = append(suiteLabels, arg...)
		case SemVerConstraints:
			suiteSemVerConstraints = append(suiteSemVerConstraints, arg...)
		case types.AroundNodeDecorator:
			aroundNodes = append(aroundNodes, arg)
		default:
			configErrors = append(configErrors, types.GinkgoErrors.UnknownTypePassedToRunSpecs(arg))
		}
	}
	exitIfErrors(configErrors)

	configErrors = types.VetConfig(flagSet, suiteConfig, reporterConfig)
	if len(configErrors) > 0 {
		fmt.Fprintf(formatter.ColorableStdErr, formatter.F("{{red}}Ginkgo detected configuration issues:{{/}}\n"))
		for _, err := range configErrors {
			fmt.Fprintf(formatter.ColorableStdErr, err.Error())
		}
		os.Exit(1)
	}

	return suiteLabels, suiteSemVerConstraints, aroundNodes
}

func getwd() (string, error) {
	if !strings.EqualFold(os.Getenv("GINKGO_PRESERVE_CACHE"), "true") {
		// Getwd calls os.Getenv("PWD"), which breaks test caching if the cache
		// is shared between two different directories with the same test code.
		return os.Getwd()
	}
	return "", nil
}

/*
PreviewSpecs walks the testing tree and produces a report without actually invoking the specs.
See http://onsi.github.io/ginkgo/#previewing-specs for more information.
*/
func PreviewSpecs(description string, args ...any) Report {
	err := global.PushClone()
	if err != nil {
		exitIfErr(err)
	}
	defer global.PopClone()

	suiteLabels, suiteSemVerConstraints, suiteAroundNodes := extractSuiteConfiguration(args)
	priorDryRun, priorParallelTotal, priorParallelProcess := suiteConfig.DryRun, suiteConfig.ParallelTotal, suiteConfig.ParallelProcess
	suiteConfig.DryRun, suiteConfig.ParallelTotal, suiteConfig.ParallelProcess = true, 1, 1
	defer func() {
		suiteConfig.DryRun, suiteConfig.ParallelTotal, suiteConfig.ParallelProcess = priorDryRun, priorParallelTotal, priorParallelProcess
	}()
	reporter := reporters.NoopReporter{}
	outputInterceptor = internal.NoopOutputInterceptor{}
	client = nil
	writer := GinkgoWriter.(*internal.Writer)

	err = global.Suite.BuildTree()
	exitIfErr(err)
	suitePath, err := getwd()
	exitIfErr(err)
	suitePath, err = filepath.Abs(suitePath)
	exitIfErr(err)

	global.Suite.Run(description, suiteLabels, suiteSemVerConstraints, suiteAroundNodes, suitePath, global.Failer, reporter, writer, outputInterceptor, interrupt_handler.NewInterruptHandler(client), client, internal.RegisterForProgressSignal, suiteConfig)

	return global.Suite.GetPreviewReport()
}

/*
Skip instructs Ginkgo to skip the current spec

You can call Skip in any Setup or Subject node closure.

For more on how to filter specs in Ginkgo see https://onsi.github.io/ginkgo/#filtering-specs
*/
func Skip(message string, callerSkip ...int) {
	skip := 0
	if len(callerSkip) > 0 {
		skip = callerSkip[0]
	}
	cl := types.NewCodeLocationWithStackTrace(skip + 1)
	global.Failer.Skip(message, cl)
	panic(types.GinkgoErrors.UncaughtGinkgoPanic(cl))
}

/*
Fail notifies Ginkgo that the current spec has failed. (Gomega will call Fail for you automatically when an assertion fails.)

Under the hood, Fail panics to end execution of the current spec.  Ginkgo will catch this panic and proceed with
the subsequent spec.  If you call Fail, or make an assertion, within a goroutine launched by your spec you must
add defer GinkgoRecover() to the goroutine to catch the panic emitted by Fail.

You can call Fail in any Setup or Subject node closure.

You can learn more about how Ginkgo manages failures here: https://onsi.github.io/ginkgo/#mental-model-how-ginkgo-handles-failure
*/
func Fail(message string, callerSkip ...int) {
	skip := 0
	if len(callerSkip) > 0 {
		skip = callerSkip[0]
	}

	cl := types.NewCodeLocationWithStackTrace(skip + 1)
	global.Failer.Fail(message, cl)
	panic(types.GinkgoErrors.UncaughtGinkgoPanic(cl))
}

/*
AbortSuite instructs Ginkgo to fail the current spec and skip all subsequent specs, thereby aborting the suite.

You can call AbortSuite in any Setup or Subject node closure.

You can learn more about how Ginkgo handles suite interruptions here: https://onsi.github.io/ginkgo/#interrupting-aborting-and-timing-out-suites
*/
func AbortSuite(message string, callerSkip ...int) {
	skip := 0
	if len(callerSkip) > 0 {
		skip = callerSkip[0]
	}

	cl := types.NewCodeLocationWithStackTrace(skip + 1)
	global.Failer.AbortSuite(message, cl)
	panic(types.GinkgoErrors.UncaughtGinkgoPanic(cl))
}

/*
ignorablePanic is used by Gomega to signal to GinkgoRecover that Goemga is handling
the error associated with this panic.  It i used when Eventually/Consistently are passed a func(g Gomega) and the resulting function launches a goroutines that makes a failed assertion.  That failed assertion is registered by Gomega and then panics.  Ordinarily the panic is captured by Gomega.  In the case of a goroutine Gomega can't capture the panic - so we piggy back on GinkgoRecover so users have a single defer GinkgoRecover() pattern to follow.  To do that we need to tell Ginkgo to ignore this panic and not register it as a panic on the global Failer.
*/
type ignorablePanic interface{ GinkgoRecoverShouldIgnoreThisPanic() }

/*
GinkgoRecover should be deferred at the top of any spawned goroutine that (may) call `Fail`
Since Gomega assertions call fail, you should throw a `defer GinkgoRecover()` at the top of any goroutine that
calls out to Gomega

Here's why: Ginkgo's `Fail` method records the failure and then panics to prevent
further assertions from running.  This panic must be recovered.  Normally, Ginkgo recovers the panic for you,
however if a panic originates on a goroutine *launched* from one of your specs there's no
way for Ginkgo to rescue the panic.  To do this, you must remember to `defer GinkgoRecover()` at the top of such a goroutine.

You can learn more about how Ginkgo manages failures here: https://onsi.github.io/ginkgo/#mental-model-how-ginkgo-handles-failure
*/
func GinkgoRecover() {
	e := recover()
	if e != nil {
		if _, ok := e.(ignorablePanic); ok {
			return
		}
		global.Failer.Panic(types.NewCodeLocationWithStackTrace(1), e)
	}
}

// pushNode is used by the various test construction DSL methods to push nodes onto the suite
// it handles returned errors, emits a detailed error message to help the user learn what they may have done wrong, then exits
func pushNode(node internal.Node, errors []error) bool {
	exitIfErrors(errors)
	exitIfErr(global.Suite.PushNode(node))
	return true
}

// NodeArgsTransformer is a hook which is called by the test construction DSL methods
// before creating the new node. If it returns any error, the test suite
// prints those errors and exits. The text and arguments can be modified,
// which includes directly changing the args slice that is passed in.
// Arguments have been flattened already, i.e. none of the entries in args is another []any.
// The result may be nested.
//
// The node type is provided for information and remains the same.
//
// The offset is valid for calling NewLocation directly in the
// implementation of TransformNodeArgs to find the location where
// the Ginkgo DSL function is called. An additional offset supplied
// by the caller via args is already included.
//
// A NodeArgsTransformer can be registered with AddTreeConstructionNodeArgsTransformer.
type NodeArgsTransformer func(nodeType types.NodeType, offset Offset, text string, args []any) (string, []any, []error)

// AddTreeConstructionNodeArgsTransformer registers a NodeArgsTransformer.
// Only nodes which get created after registering a NodeArgsTransformer
// are transformed by it. The returned function can be called to
// unregister the transformer.
//
// Both may only be called during the construction phase.
//
// If there is more than one registered transformer, then the most
// recently added ones get called first.
func AddTreeConstructionNodeArgsTransformer(transformer NodeArgsTransformer) func() {
	// This conversion could be avoided with a type alias, but type aliases make
	// developer documentation less useful.
	return internal.AddTreeConstructionNodeArgsTransformer(internal.NodeArgsTransformer(transformer))
}

/*
Describe nodes are Container nodes that allow you to organize your specs.  A Describe node's closure can contain any number of
Setup nodes (e.g. BeforeEach, AfterEach, JustBeforeEach), and Subject nodes (i.e. It).

Context and When nodes are aliases for Describe - use whichever gives your suite a better narrative flow.  It is idomatic
to Describe the behavior of an object or function and, within that Describe, outline a number of Contexts and Whens.

You can learn more at https://onsi.github.io/ginkgo/#organizing-specs-with-container-nodes
In addition, container nodes can be decorated with a variety of decorators.  You can learn more here: https://onsi.github.io/ginkgo/#decorator-reference
*/
func Describe(text string, args ...any) bool {
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeContainer, text, args...)))
}

/*
FDescribe focuses specs within the Describe block.
*/
func FDescribe(text string, args ...any) bool {
	args = append(args, internal.Focus)
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeContainer, text, args...)))
}

/*
PDescribe marks specs within the Describe block as pending.
*/
func PDescribe(text string, args ...any) bool {
	args = append(args, internal.Pending)
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeContainer, text, args...)))
}

/*
XDescribe marks specs within the Describe block as pending.

XDescribe is an alias for PDescribe
*/
var XDescribe = PDescribe

/* Context is an alias for Describe - it generates the exact same kind of Container node */
var Context, FContext, PContext, XContext = Describe, FDescribe, PDescribe, XDescribe

/* When is an alias for Describe - it generates the exact same kind of Container node with "when " as prefix for the text. */
func When(text string, args ...any) bool {
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeContainer, "when "+text, args...)))
}

/* When is an alias for Describe - it generates the exact same kind of Container node with "when " as prefix for the text. */
func FWhen(text string, args ...any) bool {
	args = append(args, internal.Focus)
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeContainer, "when "+text, args...)))
}

/* When is an alias for Describe - it generates the exact same kind of Container node */
func PWhen(text string, args ...any) bool {
	args = append(args, internal.Pending)
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeContainer, "when "+text, args...)))
}

var XWhen = PWhen

/*
It nodes are Subject nodes that contain your spec code and assertions.

Each It node corresponds to an individual Ginkgo spec.  You cannot nest any other Ginkgo nodes within an It node's closure.

You can pass It nodes bare functions (func() {}) or functions that receive a SpecContext or context.Context: func(ctx SpecContext) {} and func (ctx context.Context) {}. If the function takes a context then the It is deemed interruptible and Ginkgo will cancel the context in the event of a timeout (configured via the SpecTimeout() or NodeTimeout() decorators) or of an interrupt signal.

You can learn more at https://onsi.github.io/ginkgo/#spec-subjects-it
In addition, subject nodes can be decorated with a variety of decorators.  You can learn more here: https://onsi.github.io/ginkgo/#decorator-reference
*/
func It(text string, args ...any) bool {
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeIt, text, args...)))
}

/*
FIt allows you to focus an individual It.
*/
func FIt(text string, args ...any) bool {
	args = append(args, internal.Focus)
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeIt, text, args...)))
}

/*
PIt allows you to mark an individual It as pending.
*/
func PIt(text string, args ...any) bool {
	args = append(args, internal.Pending)
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeIt, text, args...)))
}

/*
XIt allows you to mark an individual It as pending.

XIt is an alias for PIt
*/
var XIt = PIt

/*
Specify is an alias for It - it can allow for more natural wording in some context.
*/
var Specify, FSpecify, PSpecify, XSpecify = It, FIt, PIt, XIt

/*
By allows you to better document complex Specs.

Generally you should try to keep your Its short and to the point.  This is not always possible, however,
especially in the context of integration tests that capture complex or lengthy workflows.

By allows you to document such flows.  By may be called within a Setup or Subject node (It, BeforeEach, etc...)
and will simply log the passed in text to the GinkgoWriter.  If By is handed a function it will immediately run the function.

By will also generate and attach a ReportEntry to the spec.  This will ensure that By annotations appear in Ginkgo's machine-readable reports.

Note that By does not generate a new Ginkgo node - rather it is simply syntactic sugar around GinkgoWriter and AddReportEntry
You can learn more about By here: https://onsi.github.io/ginkgo/#documenting-complex-specs-by
*/
func By(text string, callback ...func()) {
	exitIfErr(global.Suite.By(text, callback...))
}

/*
BeforeSuite nodes are suite-level Setup nodes that run just once before any specs are run.
When running in parallel, each parallel process will call BeforeSuite.

You may only register *one* BeforeSuite handler per test suite.  You typically do so in your bootstrap file at the top level.

BeforeSuite can take a func() body, or an interruptible func(SpecContext)/func(context.Context) body.

You cannot nest any other Ginkgo nodes within a BeforeSuite node's closure.
You can learn more here: https://onsi.github.io/ginkgo/#suite-setup-and-cleanup-beforesuite-and-aftersuite
*/
func BeforeSuite(body any, args ...any) bool {
	combinedArgs := []any{body}
	combinedArgs = append(combinedArgs, args...)
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeBeforeSuite, "", combinedArgs...)))
}

/*
AfterSuite nodes are suite-level Setup nodes run after all specs have finished - regardless of whether specs have passed or failed.
AfterSuite node closures always run, even if Ginkgo receives an interrupt signal (^C), in order to ensure cleanup occurs.

When running in parallel, each parallel process will call AfterSuite.

You may only register *one* AfterSuite handler per test suite.  You typically do so in your bootstrap file at the top level.

AfterSuite can take a func() body, or an interruptible func(SpecContext)/func(context.Context) body.

You cannot nest any other Ginkgo nodes within an AfterSuite node's closure.
You can learn more here: https://onsi.github.io/ginkgo/#suite-setup-and-cleanup-beforesuite-and-aftersuite
*/
func AfterSuite(body any, args ...any) bool {
	combinedArgs := []any{body}
	combinedArgs = append(combinedArgs, args...)
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeAfterSuite, "", combinedArgs...)))
}

/*
SynchronizedBeforeSuite nodes allow you to perform some of the suite setup just once - on parallel process #1 - and then pass information
from that setup to the rest of the suite setup on all processes.  This is useful for performing expensive or singleton setup once, then passing
information from that setup to all parallel processes.

SynchronizedBeforeSuite accomplishes this by taking *two* function arguments and passing data between them.
The first function is only run on parallel process #1.  The second is run on all processes, but *only* after the first function completes successfully.  The functions have the following signatures:

The first function (which only runs on process #1) can have any of the following the signatures:

	func()
	func(ctx context.Context)
	func(ctx SpecContext)
	func() []byte
	func(ctx context.Context) []byte
	func(ctx SpecContext) []byte

The byte array returned by the first function (if present) is then passed to the second function, which can have any of the following signature:

	func()
	func(ctx context.Context)
	func(ctx SpecContext)
	func(data []byte)
	func(ctx context.Context, data []byte)
	func(ctx SpecContext, data []byte)

If either function receives a context.Context/SpecContext it is considered interruptible.

You cannot nest any other Ginkgo nodes within an SynchronizedBeforeSuite node's closure.
You can learn more, and see some examples, here: https://onsi.github.io/ginkgo/#parallel-suite-setup-and-cleanup-synchronizedbeforesuite-and-synchronizedaftersuite
*/
func SynchronizedBeforeSuite(process1Body any, allProcessBody any, args ...any) bool {
	combinedArgs := []any{process1Body, allProcessBody}
	combinedArgs = append(combinedArgs, args...)

	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeSynchronizedBeforeSuite, "", combinedArgs...)))
}

/*
SynchronizedAfterSuite nodes complement the SynchronizedBeforeSuite nodes in solving the problem of splitting clean up into a piece that runs on all processes
and a piece that must only run once - on process #1.

SynchronizedAfterSuite accomplishes this by taking *two* function arguments.  The first runs on all processes.  The second runs only on parallel process #1
and *only* after all other processes have finished and exited.  This ensures that process #1, and any resources it is managing, remain alive until
all other processes are finished.  These two functions can be bare functions (func()) or interruptible (func(context.Context)/func(SpecContext))

Note that you can also use DeferCleanup() in SynchronizedBeforeSuite to accomplish similar results.

You cannot nest any other Ginkgo nodes within an SynchronizedAfterSuite node's closure.
You can learn more, and see some examples, here: https://onsi.github.io/ginkgo/#parallel-suite-setup-and-cleanup-synchronizedbeforesuite-and-synchronizedaftersuite
*/
func SynchronizedAfterSuite(allProcessBody any, process1Body any, args ...any) bool {
	combinedArgs := []any{allProcessBody, process1Body}
	combinedArgs = append(combinedArgs, args...)

	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeSynchronizedAfterSuite, "", combinedArgs...)))
}

/*
BeforeEach nodes are Setup nodes whose closures run before It node closures.  When multiple BeforeEach nodes
are defined in nested Container nodes the outermost BeforeEach node closures are run first.

BeforeEach can take a func() body, or an interruptible func(SpecContext)/func(context.Context) body.

You cannot nest any other Ginkgo nodes within a BeforeEach node's closure.
You can learn more here: https://onsi.github.io/ginkgo/#extracting-common-setup-beforeeach
*/
func BeforeEach(args ...any) bool {
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeBeforeEach, "", args...)))
}

/*
JustBeforeEach nodes are similar to BeforeEach nodes, however they are guaranteed to run *after* all BeforeEach node closures - just before the It node closure.
This can allow you to separate configuration from creation of resources for a spec.

JustBeforeEach can take a func() body, or an interruptible func(SpecContext)/func(context.Context) body.

You cannot nest any other Ginkgo nodes within a JustBeforeEach node's closure.
You can learn more and see some examples here: https://onsi.github.io/ginkgo/#separating-creation-and-configuration-justbeforeeach
*/
func JustBeforeEach(args ...any) bool {
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeJustBeforeEach, "", args...)))
}

/*
AfterEach nodes are Setup nodes whose closures run after It node closures.  When multiple AfterEach nodes
are defined in nested Container nodes the innermost AfterEach node closures are run first.

Note that you can also use DeferCleanup() in other Setup or Subject nodes to accomplish similar results.

AfterEach can take a func() body, or an interruptible func(SpecContext)/func(context.Context) body.

You cannot nest any other Ginkgo nodes within an AfterEach node's closure.
You can learn more here: https://onsi.github.io/ginkgo/#spec-cleanup-aftereach-and-defercleanup
*/
func AfterEach(args ...any) bool {
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeAfterEach, "", args...)))
}

/*
JustAfterEach nodes are similar to AfterEach nodes, however they are guaranteed to run *before* all AfterEach node closures - just after the It node closure. This can allow you to separate diagnostics collection from teardown for a spec.

JustAfterEach can take a func() body, or an interruptible func(SpecContext)/func(context.Context) body.

You cannot nest any other Ginkgo nodes within a JustAfterEach node's closure.
You can learn more and see some examples here: https://onsi.github.io/ginkgo/#separating-diagnostics-collection-and-teardown-justaftereach
*/
func JustAfterEach(args ...any) bool {
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeJustAfterEach, "", args...)))
}

/*
BeforeAll nodes are Setup nodes that can occur inside Ordered containers.  They run just once before any specs in the Ordered container run.

Multiple BeforeAll nodes can be defined in a given Ordered container however they cannot be nested inside any other container.

BeforeAll can take a func() body, or an interruptible func(SpecContext)/func(context.Context) body.

You cannot nest any other Ginkgo nodes within a BeforeAll node's closure.
You can learn more about Ordered Containers at: https://onsi.github.io/ginkgo/#ordered-containers
And you can learn more about BeforeAll at: https://onsi.github.io/ginkgo/#setup-in-ordered-containers-beforeall-and-afterall
*/
func BeforeAll(args ...any) bool {
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeBeforeAll, "", args...)))
}

/*
AfterAll nodes are Setup nodes that can occur inside Ordered containers.  They run just once after all specs in the Ordered container have run.

Multiple AfterAll nodes can be defined in a given Ordered container however they cannot be nested inside any other container.

Note that you can also use DeferCleanup() in a BeforeAll node to accomplish similar behavior.

AfterAll can take a func() body, or an interruptible func(SpecContext)/func(context.Context) body.

You cannot nest any other Ginkgo nodes within an AfterAll node's closure.
You can learn more about Ordered Containers at: https://onsi.github.io/ginkgo/#ordered-containers
And you can learn more about AfterAll at: https://onsi.github.io/ginkgo/#setup-in-ordered-containers-beforeall-and-afterall
*/
func AfterAll(args ...any) bool {
	return pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeAfterAll, "", args...)))
}

/*
DeferCleanup can be called within any Setup or Subject node to register a cleanup callback that Ginkgo will call at the appropriate time to cleanup after the spec.

DeferCleanup can be passed:
1. A function that takes no arguments and returns no values.
2. A function that returns multiple values.  `DeferCleanup` will ignore all these return values except for the last one.  If this last return value is a non-nil error `DeferCleanup` will fail the spec).
3. A function that takes a context.Context or SpecContext (and optionally returns multiple values).  The resulting cleanup node is deemed interruptible and the passed-in context will be cancelled in the event of a timeout or interrupt.
4. A function that takes arguments (and optionally returns multiple values) followed by a list of arguments to pass to the function.
5. A function that takes SpecContext and a list of arguments (and optionally returns multiple values) followed by a list of arguments to pass to the function.

For example:

	BeforeEach(func() {
	    DeferCleanup(os.Setenv, "FOO", os.GetEnv("FOO"))
	    os.Setenv("FOO", "BAR")
	})

will register a cleanup handler that will set the environment variable "FOO" to its current value (obtained by os.GetEnv("FOO")) after the spec runs and then sets the environment variable "FOO" to "BAR" for the current spec.

Similarly:

	BeforeEach(func() {
	    DeferCleanup(func(ctx SpecContext, path) {
	    	req, err := http.NewRequestWithContext(ctx, "POST", path, nil)
	    	Expect(err).NotTo(HaveOccured())
	    	_, err := http.DefaultClient.Do(req)
	    	Expect(err).NotTo(HaveOccured())
	    }, "example.com/cleanup", NodeTimeout(time.Second*3))
	})

will register a cleanup handler that will have three seconds to successfully complete a request to the specified path. Note that we do not specify a context in the list of arguments passed to DeferCleanup - only in the signature of the function we pass in.  Ginkgo will detect the requested context and supply a SpecContext when it invokes the cleanup node.  If you want to pass in your own context in addition to the Ginkgo-provided SpecContext you must specify the SpecContext as the first argument (e.g. func(ctx SpecContext, otherCtx context.Context)).

When DeferCleanup is called in BeforeEach, JustBeforeEach, It, AfterEach, or JustAfterEach the registered callback will be invoked when the spec completes (i.e. it will behave like an AfterEach node)
When DeferCleanup is called in BeforeAll or AfterAll the registered callback will be invoked when the ordered container completes (i.e. it will behave like an AfterAll node)
When DeferCleanup is called in BeforeSuite, SynchronizedBeforeSuite, AfterSuite, or SynchronizedAfterSuite the registered callback will be invoked when the suite completes (i.e. it will behave like an AfterSuite node)

Note that DeferCleanup does not represent a node but rather dynamically generates the appropriate type of cleanup node based on the context in which it is called.  As such you must call DeferCleanup within a Setup or Subject node, and not within a Container node.
You can learn more about DeferCleanup here: https://onsi.github.io/ginkgo/#cleaning-up-our-cleanup-code-defercleanup
*/
func DeferCleanup(args ...any) {
	fail := func(message string, cl types.CodeLocation) {
		global.Failer.Fail(message, cl)
	}
	pushNode(internal.NewCleanupNode(deprecationTracker, fail, args...))
}

/*
AttachProgressReporter allows you to register a function that will be called whenever Ginkgo generates a Progress Report.  The contents returned by the function will be included in the report.

**This is an experimental feature and the public-facing interface may change in a future minor version of Ginkgo**

Progress Reports are generated:
- whenever the user explicitly requests one (via `SIGINFO` or `SIGUSR1`)
- on nodes decorated  with PollProgressAfter
- on suites run with --poll-progress-after
- whenever a test times out

Ginkgo uses Progress Reports to convey the current state of the test suite, including any running goroutines.  By attaching a progress reporter you are able to supplement these reports with additional information.

# AttachProgressReporter returns a function that can be called to detach the progress reporter

You can learn more about AttachProgressReporter here: https://onsi.github.io/ginkgo/#attaching-additional-information-to-progress-reports
*/
func AttachProgressReporter(reporter func() string) func() {
	return global.Suite.AttachProgressReporter(reporter)
}
