package types

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/onsi/ginkgo/v2/formatter"
)

type GinkgoError struct {
	Heading      string
	Message      string
	DocLink      string
	CodeLocation CodeLocation
}

func (g GinkgoError) Error() string {
	out := formatter.F("{{bold}}{{red}}%s{{/}}\n", g.Heading)
	if (g.CodeLocation != CodeLocation{}) {
		contentsOfLine := strings.TrimLeft(g.CodeLocation.ContentsOfLine(), "\t ")
		if contentsOfLine != "" {
			out += formatter.F("{{light-gray}}%s{{/}}\n", contentsOfLine)
		}
		out += formatter.F("{{gray}}%s{{/}}\n", g.CodeLocation)
	}
	if g.Message != "" {
		out += formatter.Fiw(1, formatter.COLS, g.Message)
		out += "\n\n"
	}
	if g.DocLink != "" {
		out += formatter.Fiw(1, formatter.COLS, "{{bold}}Learn more at:{{/}} {{cyan}}{{underline}}http://onsi.github.io/ginkgo/#%s{{/}}\n", g.DocLink)
	}

	return out
}

type ginkgoErrors struct{}

var GinkgoErrors = ginkgoErrors{}

func (g ginkgoErrors) UncaughtGinkgoPanic(cl CodeLocation) error {
	return GinkgoError{
		Heading: "Your Test Panicked",
		Message: `When you, or your assertion library, calls Ginkgo's Fail(),
Ginkgo panics to prevent subsequent assertions from running.

Normally Ginkgo rescues this panic so you shouldn't see it.

However, if you make an assertion in a goroutine, Ginkgo can't capture the panic.
To circumvent this, you should call

	defer GinkgoRecover()

at the top of the goroutine that caused this panic.

Alternatively, you may have made an assertion outside of a Ginkgo
leaf node (e.g. in a container node or some out-of-band function) - please move your assertion to
an appropriate Ginkgo node (e.g. a BeforeSuite, BeforeEach, It, etc...).`,
		DocLink:      "mental-model-how-ginkgo-handles-failure",
		CodeLocation: cl,
	}
}

func (g ginkgoErrors) RerunningSuite() error {
	return GinkgoError{
		Heading: "Rerunning Suite",
		Message: formatter.F(`It looks like you are calling RunSpecs more than once. Ginkgo does not support rerunning suites.  If you want to rerun a suite try {{bold}}ginkgo --repeat=N{{/}} or {{bold}}ginkgo --until-it-fails{{/}}`),
		DocLink: "repeating-spec-runs-and-managing-flaky-specs",
	}
}

/* Tree construction errors */

func (g ginkgoErrors) PushingNodeInRunPhase(nodeType NodeType, cl CodeLocation) error {
	return GinkgoError{
		Heading: "Ginkgo detected an issue with your spec structure",
		Message: formatter.F(
			`It looks like you are trying to add a {{bold}}[%s]{{/}} node
to the Ginkgo spec tree in a leaf node {{bold}}after{{/}} the specs started running.

To enable randomization and parallelization Ginkgo requires the spec tree
to be fully constructed up front.  In practice, this means that you can
only create nodes like {{bold}}[%s]{{/}} at the top-level or within the
body of a {{bold}}Describe{{/}}, {{bold}}Context{{/}}, or {{bold}}When{{/}}.`, nodeType, nodeType),
		CodeLocation: cl,
		DocLink:      "mental-model-how-ginkgo-traverses-the-spec-hierarchy",
	}
}

func (g ginkgoErrors) CaughtPanicDuringABuildPhase(caughtPanic interface{}, cl CodeLocation) error {
	return GinkgoError{
		Heading: "Assertion or Panic detected during tree construction",
		Message: formatter.F(
			`Ginkgo detected a panic while constructing the spec tree.
You may be trying to make an assertion in the body of a container node
(i.e. {{bold}}Describe{{/}}, {{bold}}Context{{/}}, or {{bold}}When{{/}}).

Please ensure all assertions are inside leaf nodes such as {{bold}}BeforeEach{{/}},
{{bold}}It{{/}}, etc.

{{bold}}Here's the content of the panic that was caught:{{/}}
%v`, caughtPanic),
		CodeLocation: cl,
		DocLink:      "no-assertions-in-container-nodes",
	}
}

func (g ginkgoErrors) SuiteNodeInNestedContext(nodeType NodeType, cl CodeLocation) error {
	docLink := "suite-setup-and-cleanup-beforesuite-and-aftersuite"
	if nodeType.Is(NodeTypeReportBeforeSuite | NodeTypeReportAfterSuite) {
		docLink = "reporting-nodes---reportbeforesuite-and-reportaftersuite"
	}

	return GinkgoError{
		Heading: "Ginkgo detected an issue with your spec structure",
		Message: formatter.F(
			`It looks like you are trying to add a {{bold}}[%s]{{/}} node within a container node.

{{bold}}%s{{/}} can only be called at the top level.`, nodeType, nodeType),
		CodeLocation: cl,
		DocLink:      docLink,
	}
}

func (g ginkgoErrors) SuiteNodeDuringRunPhase(nodeType NodeType, cl CodeLocation) error {
	docLink := "suite-setup-and-cleanup-beforesuite-and-aftersuite"
	if nodeType.Is(NodeTypeReportBeforeSuite | NodeTypeReportAfterSuite) {
		docLink = "reporting-nodes---reportbeforesuite-and-reportaftersuite"
	}

	return GinkgoError{
		Heading: "Ginkgo detected an issue with your spec structure",
		Message: formatter.F(
			`It looks like you are trying to add a {{bold}}[%s]{{/}} node within a leaf node after the spec started running.

{{bold}}%s{{/}} can only be called at the top level.`, nodeType, nodeType),
		CodeLocation: cl,
		DocLink:      docLink,
	}
}

func (g ginkgoErrors) MultipleBeforeSuiteNodes(nodeType NodeType, cl CodeLocation, earlierNodeType NodeType, earlierCodeLocation CodeLocation) error {
	return ginkgoErrorMultipleSuiteNodes("setup", nodeType, cl, earlierNodeType, earlierCodeLocation)
}

func (g ginkgoErrors) MultipleAfterSuiteNodes(nodeType NodeType, cl CodeLocation, earlierNodeType NodeType, earlierCodeLocation CodeLocation) error {
	return ginkgoErrorMultipleSuiteNodes("teardown", nodeType, cl, earlierNodeType, earlierCodeLocation)
}

func ginkgoErrorMultipleSuiteNodes(setupOrTeardown string, nodeType NodeType, cl CodeLocation, earlierNodeType NodeType, earlierCodeLocation CodeLocation) error {
	return GinkgoError{
		Heading: "Ginkgo detected an issue with your spec structure",
		Message: formatter.F(
			`It looks like you are trying to add a {{bold}}[%s]{{/}} node but
you already have a {{bold}}[%s]{{/}} node defined at: {{gray}}%s{{/}}.

Ginkgo only allows you to define one suite %s node.`, nodeType, earlierNodeType, earlierCodeLocation, setupOrTeardown),
		CodeLocation: cl,
		DocLink:      "suite-setup-and-cleanup-beforesuite-and-aftersuite",
	}
}

/* Decorator errors */
func (g ginkgoErrors) InvalidDecoratorForNodeType(cl CodeLocation, nodeType NodeType, decorator string) error {
	return GinkgoError{
		Heading:      "Invalid Decorator",
		Message:      formatter.F(`[%s] node cannot be passed a(n) '%s' decorator`, nodeType, decorator),
		CodeLocation: cl,
		DocLink:      "node-decorators-overview",
	}
}

func (g ginkgoErrors) InvalidDeclarationOfFocusedAndPending(cl CodeLocation, nodeType NodeType) error {
	return GinkgoError{
		Heading:      "Invalid Combination of Decorators: Focused and Pending",
		Message:      formatter.F(`[%s] node was decorated with both Focus and Pending.  At most one is allowed.`, nodeType),
		CodeLocation: cl,
		DocLink:      "node-decorators-overview",
	}
}

func (g ginkgoErrors) InvalidDeclarationOfFlakeAttemptsAndMustPassRepeatedly(cl CodeLocation, nodeType NodeType) error {
	return GinkgoError{
		Heading:      "Invalid Combination of Decorators: FlakeAttempts and MustPassRepeatedly",
		Message:      formatter.F(`[%s] node was decorated with both FlakeAttempts and MustPassRepeatedly. At most one is allowed.`, nodeType),
		CodeLocation: cl,
		DocLink:      "node-decorators-overview",
	}
}

func (g ginkgoErrors) UnknownDecorator(cl CodeLocation, nodeType NodeType, decorator interface{}) error {
	return GinkgoError{
		Heading:      "Unknown Decorator",
		Message:      formatter.F(`[%s] node was passed an unknown decorator: '%#v'`, nodeType, decorator),
		CodeLocation: cl,
		DocLink:      "node-decorators-overview",
	}
}

func (g ginkgoErrors) InvalidBodyTypeForContainer(t reflect.Type, cl CodeLocation, nodeType NodeType) error {
	return GinkgoError{
		Heading:      "Invalid Function",
		Message:      formatter.F(`[%s] node must be passed {{bold}}func(){{/}} - i.e. functions that take nothing and return nothing.  You passed {{bold}}%s{{/}} instead.`, nodeType, t),
		CodeLocation: cl,
		DocLink:      "node-decorators-overview",
	}
}

func (g ginkgoErrors) InvalidBodyType(t reflect.Type, cl CodeLocation, nodeType NodeType) error {
	mustGet := "{{bold}}func(){{/}}, {{bold}}func(ctx SpecContext){{/}}, or {{bold}}func(ctx context.Context){{/}}"
	if nodeType.Is(NodeTypeContainer) {
		mustGet = "{{bold}}func(){{/}}"
	}
	return GinkgoError{
		Heading: "Invalid Function",
		Message: formatter.F(`[%s] node must be passed `+mustGet+`.
You passed {{bold}}%s{{/}} instead.`, nodeType, t),
		CodeLocation: cl,
		DocLink:      "node-decorators-overview",
	}
}

func (g ginkgoErrors) InvalidBodyTypeForSynchronizedBeforeSuiteProc1(t reflect.Type, cl CodeLocation) error {
	mustGet := "{{bold}}func() []byte{{/}}, {{bold}}func(ctx SpecContext) []byte{{/}}, or {{bold}}func(ctx context.Context) []byte{{/}}, {{bold}}func(){{/}}, {{bold}}func(ctx SpecContext){{/}}, or {{bold}}func(ctx context.Context){{/}}"
	return GinkgoError{
		Heading: "Invalid Function",
		Message: formatter.F(`[SynchronizedBeforeSuite] node must be passed `+mustGet+` for its first function.
You passed {{bold}}%s{{/}} instead.`, t),
		CodeLocation: cl,
		DocLink:      "node-decorators-overview",
	}
}

func (g ginkgoErrors) InvalidBodyTypeForSynchronizedBeforeSuiteAllProcs(t reflect.Type, cl CodeLocation) error {
	mustGet := "{{bold}}func(){{/}}, {{bold}}func(ctx SpecContext){{/}}, or {{bold}}func(ctx context.Context){{/}}, {{bold}}func([]byte){{/}}, {{bold}}func(ctx SpecContext, []byte){{/}}, or {{bold}}func(ctx context.Context, []byte){{/}}"
	return GinkgoError{
		Heading: "Invalid Function",
		Message: formatter.F(`[SynchronizedBeforeSuite] node must be passed `+mustGet+` for its second function.
You passed {{bold}}%s{{/}} instead.`, t),
		CodeLocation: cl,
		DocLink:      "node-decorators-overview",
	}
}

func (g ginkgoErrors) MultipleBodyFunctions(cl CodeLocation, nodeType NodeType) error {
	return GinkgoError{
		Heading:      "Multiple Functions",
		Message:      formatter.F(`[%s] node must be passed a single function - but more than one was passed in.`, nodeType),
		CodeLocation: cl,
		DocLink:      "node-decorators-overview",
	}
}

func (g ginkgoErrors) MissingBodyFunction(cl CodeLocation, nodeType NodeType) error {
	return GinkgoError{
		Heading:      "Missing Functions",
		Message:      formatter.F(`[%s] node must be passed a single function - but none was passed in.`, nodeType),
		CodeLocation: cl,
		DocLink:      "node-decorators-overview",
	}
}

func (g ginkgoErrors) InvalidTimeoutOrGracePeriodForNonContextNode(cl CodeLocation, nodeType NodeType) error {
	return GinkgoError{
		Heading:      "Invalid NodeTimeout SpecTimeout, or GracePeriod",
		Message:      formatter.F(`[%s] was passed NodeTimeout, SpecTimeout, or GracePeriod but does not have a callback that accepts a {{bold}}SpecContext{{/}} or {{bold}}context.Context{{/}}.  You must accept a context to enable timeouts and grace periods`, nodeType),
		CodeLocation: cl,
		DocLink:      "spec-timeouts-and-interruptible-nodes",
	}
}

func (g ginkgoErrors) InvalidTimeoutOrGracePeriodForNonContextCleanupNode(cl CodeLocation) error {
	return GinkgoError{
		Heading:      "Invalid NodeTimeout SpecTimeout, or GracePeriod",
		Message:      formatter.F(`[DeferCleanup] was passed NodeTimeout or GracePeriod but does not have a callback that accepts a {{bold}}SpecContext{{/}} or {{bold}}context.Context{{/}}.  You must accept a context to enable timeouts and grace periods`),
		CodeLocation: cl,
		DocLink:      "spec-timeouts-and-interruptible-nodes",
	}
}

/* Ordered Container errors */
func (g ginkgoErrors) InvalidSerialNodeInNonSerialOrderedContainer(cl CodeLocation, nodeType NodeType) error {
	return GinkgoError{
		Heading:      "Invalid Serial Node in Non-Serial Ordered Container",
		Message:      formatter.F(`[%s] node was decorated with Serial but occurs in an Ordered container that is not marked Serial.  Move the Serial decorator to the outer-most Ordered container to mark all ordered specs within the container as serial.`, nodeType),
		CodeLocation: cl,
		DocLink:      "node-decorators-overview",
	}
}

func (g ginkgoErrors) SetupNodeNotInOrderedContainer(cl CodeLocation, nodeType NodeType) error {
	return GinkgoError{
		Heading:      "Setup Node not in Ordered Container",
		Message:      fmt.Sprintf("[%s] setup nodes must appear inside an Ordered container.  They cannot be nested within other containers, even containers in an ordered container.", nodeType),
		CodeLocation: cl,
		DocLink:      "ordered-containers",
	}
}

func (g ginkgoErrors) InvalidContinueOnFailureDecoration(cl CodeLocation) error {
	return GinkgoError{
		Heading:      "ContinueOnFailure not decorating an outermost Ordered Container",
		Message:      "ContinueOnFailure can only decorate an Ordered container, and this Ordered container must be the outermost Ordered container.",
		CodeLocation: cl,
		DocLink:      "ordered-containers",
	}
}

/* DeferCleanup errors */
func (g ginkgoErrors) DeferCleanupInvalidFunction(cl CodeLocation) error {
	return GinkgoError{
		Heading:      "DeferCleanup requires a valid function",
		Message:      "You must pass DeferCleanup a function to invoke.  This function must return zero or one values - if it does return, it must return an error.  The function can take arbitrarily many arguments and you should provide these to DeferCleanup to pass along to the function.",
		CodeLocation: cl,
		DocLink:      "cleaning-up-our-cleanup-code-defercleanup",
	}
}

func (g ginkgoErrors) PushingCleanupNodeDuringTreeConstruction(cl CodeLocation) error {
	return GinkgoError{
		Heading:      "DeferCleanup must be called inside a setup or subject node",
		Message:      "You must call DeferCleanup inside a setup node (e.g. BeforeEach, BeforeSuite, AfterAll...) or a subject node (i.e. It).  You can't call DeferCleanup at the top-level or in a container node - use the After* family of setup nodes instead.",
		CodeLocation: cl,
		DocLink:      "cleaning-up-our-cleanup-code-defercleanup",
	}
}

func (g ginkgoErrors) PushingCleanupInReportingNode(cl CodeLocation, nodeType NodeType) error {
	return GinkgoError{
		Heading:      fmt.Sprintf("DeferCleanup cannot be called in %s", nodeType),
		Message:      "Please inline your cleanup code - Ginkgo won't run cleanup code after a Reporting node.",
		CodeLocation: cl,
		DocLink:      "cleaning-up-our-cleanup-code-defercleanup",
	}
}

func (g ginkgoErrors) PushingCleanupInCleanupNode(cl CodeLocation) error {
	return GinkgoError{
		Heading:      "DeferCleanup cannot be called in a DeferCleanup callback",
		Message:      "Please inline your cleanup code - Ginkgo doesn't let you call DeferCleanup from within DeferCleanup",
		CodeLocation: cl,
		DocLink:      "cleaning-up-our-cleanup-code-defercleanup",
	}
}

/* ReportEntry errors */
func (g ginkgoErrors) TooManyReportEntryValues(cl CodeLocation, arg interface{}) error {
	return GinkgoError{
		Heading:      "Too Many ReportEntry Values",
		Message:      formatter.F(`{{bold}}AddGinkgoReport{{/}} can only be given one value. Got unexpected value: %#v`, arg),
		CodeLocation: cl,
		DocLink:      "attaching-data-to-reports",
	}
}

func (g ginkgoErrors) AddReportEntryNotDuringRunPhase(cl CodeLocation) error {
	return GinkgoError{
		Heading:      "Ginkgo detected an issue with your spec structure",
		Message:      formatter.F(`It looks like you are calling {{bold}}AddGinkgoReport{{/}} outside of a running spec.  Make sure you call {{bold}}AddGinkgoReport{{/}} inside a runnable node such as It or BeforeEach and not inside the body of a container such as Describe or Context.`),
		CodeLocation: cl,
		DocLink:      "attaching-data-to-reports",
	}
}

/* By errors */
func (g ginkgoErrors) ByNotDuringRunPhase(cl CodeLocation) error {
	return GinkgoError{
		Heading:      "Ginkgo detected an issue with your spec structure",
		Message:      formatter.F(`It looks like you are calling {{bold}}By{{/}} outside of a running spec.  Make sure you call {{bold}}By{{/}} inside a runnable node such as It or BeforeEach and not inside the body of a container such as Describe or Context.`),
		CodeLocation: cl,
		DocLink:      "documenting-complex-specs-by",
	}
}

/* FileFilter and SkipFilter errors */
func (g ginkgoErrors) InvalidFileFilter(filter string) error {
	return GinkgoError{
		Heading: "Invalid File Filter",
		Message: fmt.Sprintf(`The provided file filter: "%s" is invalid.  File filters must have the format "file", "file:lines" where "file" is a regular expression that will match against the file path and lines is a comma-separated list of integers (e.g. file:1,5,7) or line-ranges (e.g. file:1-3,5-9) or both (e.g. file:1,5-9)`, filter),
		DocLink: "filtering-specs",
	}
}

func (g ginkgoErrors) InvalidFileFilterRegularExpression(filter string, err error) error {
	return GinkgoError{
		Heading: "Invalid File Filter Regular Expression",
		Message: fmt.Sprintf(`The provided file filter: "%s" included an invalid regular expression.  regexp.Compile error: %s`, filter, err),
		DocLink: "filtering-specs",
	}
}

/* Label Errors */
func (g ginkgoErrors) SyntaxErrorParsingLabelFilter(input string, location int, error string) error {
	var message string
	if location >= 0 {
		for i, r := range input {
			if i == location {
				message += "{{red}}{{bold}}{{underline}}"
			}
			message += string(r)
			if i == location {
				message += "{{/}}"
			}
		}
	} else {
		message = input
	}
	message += "\n" + error
	return GinkgoError{
		Heading: "Syntax Error Parsing Label Filter",
		Message: message,
		DocLink: "spec-labels",
	}
}

func (g ginkgoErrors) InvalidLabel(label string, cl CodeLocation) error {
	return GinkgoError{
		Heading:      "Invalid Label",
		Message:      fmt.Sprintf("'%s' is an invalid label.  Labels cannot contain of the following characters: '&|!,()/'", label),
		CodeLocation: cl,
		DocLink:      "spec-labels",
	}
}

func (g ginkgoErrors) InvalidEmptyLabel(cl CodeLocation) error {
	return GinkgoError{
		Heading:      "Invalid Empty Label",
		Message:      "Labels cannot be empty",
		CodeLocation: cl,
		DocLink:      "spec-labels",
	}
}

/* Table errors */
func (g ginkgoErrors) MultipleEntryBodyFunctionsForTable(cl CodeLocation) error {
	return GinkgoError{
		Heading:      "DescribeTable passed multiple functions",
		Message:      "It looks like you are passing multiple functions into DescribeTable.  Only one function can be passed in.  This function will be called for each Entry in the table.",
		CodeLocation: cl,
		DocLink:      "table-specs",
	}
}

func (g ginkgoErrors) InvalidEntryDescription(cl CodeLocation) error {
	return GinkgoError{
		Heading:      "Invalid Entry description",
		Message:      "Entry description functions must be a string, a function that accepts the entry parameters and returns a string, or nil.",
		CodeLocation: cl,
		DocLink:      "table-specs",
	}
}

func (g ginkgoErrors) MissingParametersForTableFunction(cl CodeLocation) error {
	return GinkgoError{
		Heading:      fmt.Sprintf("No parameters have been passed to the Table Function"),
		Message:      fmt.Sprintf("The Table Function expected at least 1 parameter"),
		CodeLocation: cl,
		DocLink:      "table-specs",
	}
}

func (g ginkgoErrors) IncorrectParameterTypeForTable(i int, name string, cl CodeLocation) error {
	return GinkgoError{
		Heading:      "DescribeTable passed incorrect parameter type",
		Message:      fmt.Sprintf("Parameter #%d passed to DescribeTable is of incorrect type <%s>", i, name),
		CodeLocation: cl,
		DocLink:      "table-specs",
	}
}

func (g ginkgoErrors) TooFewParametersToTableFunction(expected, actual int, kind string, cl CodeLocation) error {
	return GinkgoError{
		Heading:      fmt.Sprintf("Too few parameters passed in to %s", kind),
		Message:      fmt.Sprintf("The %s expected %d parameters but you passed in %d", kind, expected, actual),
		CodeLocation: cl,
		DocLink:      "table-specs",
	}
}

func (g ginkgoErrors) TooManyParametersToTableFunction(expected, actual int, kind string, cl CodeLocation) error {
	return GinkgoError{
		Heading:      fmt.Sprintf("Too many parameters passed in to %s", kind),
		Message:      fmt.Sprintf("The %s expected %d parameters but you passed in %d", kind, expected, actual),
		CodeLocation: cl,
		DocLink:      "table-specs",
	}
}

func (g ginkgoErrors) IncorrectParameterTypeToTableFunction(i int, expected, actual reflect.Type, kind string, cl CodeLocation) error {
	return GinkgoError{
		Heading:      fmt.Sprintf("Incorrect parameters type passed to %s", kind),
		Message:      fmt.Sprintf("The %s expected parameter #%d to be of type <%s> but you passed in <%s>", kind, i, expected, actual),
		CodeLocation: cl,
		DocLink:      "table-specs",
	}
}

func (g ginkgoErrors) IncorrectVariadicParameterTypeToTableFunction(expected, actual reflect.Type, kind string, cl CodeLocation) error {
	return GinkgoError{
		Heading:      fmt.Sprintf("Incorrect parameters type passed to %s", kind),
		Message:      fmt.Sprintf("The %s expected its variadic parameters to be of type <%s> but you passed in <%s>", kind, expected, actual),
		CodeLocation: cl,
		DocLink:      "table-specs",
	}
}

/* Parallel Synchronization errors */

func (g ginkgoErrors) AggregatedReportUnavailableDueToNodeDisappearing() error {
	return GinkgoError{
		Heading: "Test Report unavailable because a Ginkgo parallel process disappeared",
		Message: "The aggregated report could not be fetched for a ReportAfterSuite node.  A Ginkgo parallel process disappeared before it could finish reporting.",
	}
}

func (g ginkgoErrors) SynchronizedBeforeSuiteFailedOnProc1() error {
	return GinkgoError{
		Heading: "SynchronizedBeforeSuite failed on Ginkgo parallel process #1",
		Message: "The first SynchronizedBeforeSuite function running on Ginkgo parallel process #1 failed.  This suite will now abort.",
	}
}

func (g ginkgoErrors) SynchronizedBeforeSuiteDisappearedOnProc1() error {
	return GinkgoError{
		Heading: "Process #1 disappeared before SynchronizedBeforeSuite could report back",
		Message: "Ginkgo parallel process #1 disappeared before the first SynchronizedBeforeSuite function completed.  This suite will now abort.",
	}
}

/* Configuration errors */

func (g ginkgoErrors) UnknownTypePassedToRunSpecs(value interface{}) error {
	return GinkgoError{
		Heading: "Unknown Type passed to RunSpecs",
		Message: fmt.Sprintf("RunSpecs() accepts labels, and configuration of type types.SuiteConfig and/or types.ReporterConfig.\n You passed in: %v", value),
	}
}

var sharedParallelErrorMessage = "It looks like you are trying to run specs in parallel with go test.\nThis is unsupported and you should use the ginkgo CLI instead."

func (g ginkgoErrors) InvalidParallelTotalConfiguration() error {
	return GinkgoError{
		Heading: "-ginkgo.parallel.total must be >= 1",
		Message: sharedParallelErrorMessage,
		DocLink: "spec-parallelization",
	}
}

func (g ginkgoErrors) InvalidParallelProcessConfiguration() error {
	return GinkgoError{
		Heading: "-ginkgo.parallel.process is one-indexed and must be <= ginkgo.parallel.total",
		Message: sharedParallelErrorMessage,
		DocLink: "spec-parallelization",
	}
}

func (g ginkgoErrors) MissingParallelHostConfiguration() error {
	return GinkgoError{
		Heading: "-ginkgo.parallel.host is missing",
		Message: sharedParallelErrorMessage,
		DocLink: "spec-parallelization",
	}
}

func (g ginkgoErrors) UnreachableParallelHost(host string) error {
	return GinkgoError{
		Heading: "Could not reach ginkgo.parallel.host:" + host,
		Message: sharedParallelErrorMessage,
		DocLink: "spec-parallelization",
	}
}

func (g ginkgoErrors) DryRunInParallelConfiguration() error {
	return GinkgoError{
		Heading: "Ginkgo only performs -dryRun in serial mode.",
		Message: "Please try running ginkgo -dryRun again, but without -p or -procs to ensure the suite is running in series.",
	}
}

func (g ginkgoErrors) GracePeriodCannotBeZero() error {
	return GinkgoError{
		Heading: "Ginkgo requires a positive --grace-period.",
		Message: "Please set --grace-period to a positive duration.  The default is 30s.",
	}
}

func (g ginkgoErrors) ConflictingVerbosityConfiguration() error {
	return GinkgoError{
		Heading: "Conflicting reporter verbosity settings.",
		Message: "You can't set more than one of -v, -vv and --succinct.  Please pick one!",
	}
}

func (g ginkgoErrors) InvalidOutputInterceptorModeConfiguration(value string) error {
	return GinkgoError{
		Heading: fmt.Sprintf("Invalid value '%s' for --output-interceptor-mode.", value),
		Message: "You must choose one of 'dup', 'swap', or 'none'.",
	}
}

func (g ginkgoErrors) InvalidGoFlagCount() error {
	return GinkgoError{
		Heading: "Use of go test -count",
		Message: "Ginkgo does not support using go test -count to rerun suites.  Only -count=1 is allowed.  To repeat suite runs, please use the ginkgo cli and `ginkgo -until-it-fails` or `ginkgo -repeat=N`.",
	}
}

func (g ginkgoErrors) InvalidGoFlagParallel() error {
	return GinkgoError{
		Heading: "Use of go test -parallel",
		Message: "Go test's implementation of parallelization does not actually parallelize Ginkgo specs.  Please use the ginkgo cli and `ginkgo -p` or `ginkgo -procs=N` instead.",
	}
}

func (g ginkgoErrors) BothRepeatAndUntilItFails() error {
	return GinkgoError{
		Heading: "--repeat and --until-it-fails are both set",
		Message: "--until-it-fails directs Ginkgo to rerun specs indefinitely until they fail.  --repeat directs Ginkgo to rerun specs a set number of times.  You can't set both... which would you like?",
	}
}

/* Stack-Trace parsing errors */

func (g ginkgoErrors) FailedToParseStackTrace(message string) error {
	return GinkgoError{
		Heading: "Failed to Parse Stack Trace",
		Message: message,
	}
}
