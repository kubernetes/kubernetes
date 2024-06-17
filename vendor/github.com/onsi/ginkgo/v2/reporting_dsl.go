package ginkgo

import (
	"fmt"
	"strings"

	"github.com/onsi/ginkgo/v2/internal"
	"github.com/onsi/ginkgo/v2/internal/global"
	"github.com/onsi/ginkgo/v2/reporters"
	"github.com/onsi/ginkgo/v2/types"
)

/*
Report represents the report for a Suite.
It is documented here: https://pkg.go.dev/github.com/onsi/ginkgo/v2/types#Report
*/
type Report = types.Report

/*
Report represents the report for a Spec.
It is documented here: https://pkg.go.dev/github.com/onsi/ginkgo/v2/types#SpecReport
*/
type SpecReport = types.SpecReport

/*
CurrentSpecReport returns information about the current running spec.
The returned object is a types.SpecReport which includes helper methods
to make extracting information about the spec easier.

You can learn more about SpecReport here: https://pkg.go.dev/github.com/onsi/ginkgo/types#SpecReport
You can learn more about CurrentSpecReport() here: https://onsi.github.io/ginkgo/#getting-a-report-for-the-current-spec
*/
func CurrentSpecReport() SpecReport {
	return global.Suite.CurrentSpecReport()
}

/*
	ReportEntryVisibility governs the visibility of ReportEntries in Ginkgo's console reporter

- ReportEntryVisibilityAlways: the default behavior - the ReportEntry is always emitted.
- ReportEntryVisibilityFailureOrVerbose: the ReportEntry is only emitted if the spec fails or if the tests are run with -v (similar to GinkgoWriters behavior).
- ReportEntryVisibilityNever: the ReportEntry is never emitted though it appears in any generated machine-readable reports (e.g. by setting `--json-report`).

You can learn more about Report Entries here: https://onsi.github.io/ginkgo/#attaching-data-to-reports
*/
type ReportEntryVisibility = types.ReportEntryVisibility

const ReportEntryVisibilityAlways, ReportEntryVisibilityFailureOrVerbose, ReportEntryVisibilityNever = types.ReportEntryVisibilityAlways, types.ReportEntryVisibilityFailureOrVerbose, types.ReportEntryVisibilityNever

/*
AddReportEntry generates and adds a new ReportEntry to the current spec's SpecReport.
It can take any of the following arguments:
  - A single arbitrary object to attach as the Value of the ReportEntry.  This object will be included in any generated reports and will be emitted to the console when the report is emitted.
  - A ReportEntryVisibility enum to control the visibility of the ReportEntry
  - An Offset or CodeLocation decoration to control the reported location of the ReportEntry

If the Value object implements `fmt.Stringer`, it's `String()` representation is used when emitting to the console.

AddReportEntry() must be called within a Subject or Setup node - not in a Container node.

You can learn more about Report Entries here: https://onsi.github.io/ginkgo/#attaching-data-to-reports
*/
func AddReportEntry(name string, args ...interface{}) {
	cl := types.NewCodeLocation(1)
	reportEntry, err := internal.NewReportEntry(name, cl, args...)
	if err != nil {
		Fail(fmt.Sprintf("Failed to generate Report Entry:\n%s", err.Error()), 1)
	}
	err = global.Suite.AddReportEntry(reportEntry)
	if err != nil {
		Fail(fmt.Sprintf("Failed to add Report Entry:\n%s", err.Error()), 1)
	}
}

/*
ReportBeforeEach nodes are run for each spec, even if the spec is skipped or pending.  ReportBeforeEach nodes take a function that
receives a SpecReport or both SpecContext and Report for interruptible behavior. They are called before the spec starts.

Example:

	ReportBeforeEach(func(report SpecReport) { // process report  })
	ReportBeforeEach(func(ctx SpecContext, report SpecReport) {
		// process report
	}), NodeTimeout(1 * time.Minute))

You cannot nest any other Ginkgo nodes within a ReportBeforeEach node's closure.
You can learn more about ReportBeforeEach here: https://onsi.github.io/ginkgo/#generating-reports-programmatically

You can learn about interruptible nodes here: https://onsi.github.io/ginkgo/#spec-timeouts-and-interruptible-nodes
*/
func ReportBeforeEach(body any, args ...any) bool {
	combinedArgs := []interface{}{body}
	combinedArgs = append(combinedArgs, args...)

	return pushNode(internal.NewNode(deprecationTracker, types.NodeTypeReportBeforeEach, "", combinedArgs...))
}

/*
ReportAfterEach nodes are run for each spec, even if the spec is skipped or pending.
ReportAfterEach nodes take a function that receives a SpecReport or both SpecContext and Report for interruptible behavior.
They are called after the spec has completed and receive the final report for the spec.

Example:

	ReportAfterEach(func(report SpecReport) { // process report  })
	ReportAfterEach(func(ctx SpecContext, report SpecReport) {
		// process report
	}), NodeTimeout(1 * time.Minute))

You cannot nest any other Ginkgo nodes within a ReportAfterEach node's closure.
You can learn more about ReportAfterEach here: https://onsi.github.io/ginkgo/#generating-reports-programmatically

You can learn about interruptible nodes here: https://onsi.github.io/ginkgo/#spec-timeouts-and-interruptible-nodes
*/
func ReportAfterEach(body any, args ...any) bool {
	combinedArgs := []interface{}{body}
	combinedArgs = append(combinedArgs, args...)

	return pushNode(internal.NewNode(deprecationTracker, types.NodeTypeReportAfterEach, "", combinedArgs...))
}

/*
ReportBeforeSuite nodes are run at the beginning of the suite.  ReportBeforeSuite nodes take a function
that can either receive Report or both SpecContext and Report for interruptible behavior.

Example Usage:

	ReportBeforeSuite(func(r Report) { // process report })
	ReportBeforeSuite(func(ctx SpecContext, r Report) {
		// process report
	}, NodeTimeout(1 * time.Minute))

They are called at the beginning of the suite, before any specs have run and any BeforeSuite or SynchronizedBeforeSuite nodes, and are passed in the initial report for the suite.
ReportBeforeSuite nodes must be created at the top-level (i.e. not nested in a Context/Describe/When node)

# When running in parallel, Ginkgo ensures that only one of the parallel nodes runs the ReportBeforeSuite

You cannot nest any other Ginkgo nodes within a ReportAfterSuite node's closure.
You can learn more about ReportAfterSuite here: https://onsi.github.io/ginkgo/#generating-reports-programmatically

You can learn more about Ginkgo's reporting infrastructure, including generating reports with the CLI here: https://onsi.github.io/ginkgo/#generating-machine-readable-reports

You can learn about interruptible nodes here: https://onsi.github.io/ginkgo/#spec-timeouts-and-interruptible-nodes
*/
func ReportBeforeSuite(body any, args ...any) bool {
	combinedArgs := []interface{}{body}
	combinedArgs = append(combinedArgs, args...)
	return pushNode(internal.NewNode(deprecationTracker, types.NodeTypeReportBeforeSuite, "", combinedArgs...))
}

/*
ReportAfterSuite nodes are run at the end of the suite. ReportAfterSuite nodes execute at the suite's conclusion,
and accept a function that can either receive Report or both SpecContext and Report for interruptible behavior.

Example Usage:

	ReportAfterSuite("Non-interruptible ReportAfterSuite", func(r Report) { // process report })
	ReportAfterSuite("Interruptible ReportAfterSuite", func(ctx SpecContext, r Report) {
		// process report
	}, NodeTimeout(1 * time.Minute))

They are called at the end of the suite, after all specs have run and any AfterSuite or SynchronizedAfterSuite nodes, and are passed in the final report for the suite.
ReportAfterSuite nodes must be created at the top-level (i.e. not nested in a Context/Describe/When node)

When running in parallel, Ginkgo ensures that only one of the parallel nodes runs the ReportAfterSuite and that it is passed a report that is aggregated across
all parallel nodes

In addition to using ReportAfterSuite to programmatically generate suite reports, you can also generate JSON, JUnit, and Teamcity formatted reports using the --json-report, --junit-report, and --teamcity-report ginkgo CLI flags.

You cannot nest any other Ginkgo nodes within a ReportAfterSuite node's closure.
You can learn more about ReportAfterSuite here: https://onsi.github.io/ginkgo/#generating-reports-programmatically

You can learn more about Ginkgo's reporting infrastructure, including generating reports with the CLI here: https://onsi.github.io/ginkgo/#generating-machine-readable-reports

You can learn about interruptible nodes here: https://onsi.github.io/ginkgo/#spec-timeouts-and-interruptible-nodes
*/
func ReportAfterSuite(text string, body any, args ...interface{}) bool {
	combinedArgs := []interface{}{body}
	combinedArgs = append(combinedArgs, args...)
	return pushNode(internal.NewNode(deprecationTracker, types.NodeTypeReportAfterSuite, text, combinedArgs...))
}

func registerReportAfterSuiteNodeForAutogeneratedReports(reporterConfig types.ReporterConfig) {
	body := func(report Report) {
		if reporterConfig.JSONReport != "" {
			err := reporters.GenerateJSONReport(report, reporterConfig.JSONReport)
			if err != nil {
				Fail(fmt.Sprintf("Failed to generate JSON report:\n%s", err.Error()))
			}
		}
		if reporterConfig.JUnitReport != "" {
			err := reporters.GenerateJUnitReport(report, reporterConfig.JUnitReport)
			if err != nil {
				Fail(fmt.Sprintf("Failed to generate JUnit report:\n%s", err.Error()))
			}
		}
		if reporterConfig.TeamcityReport != "" {
			err := reporters.GenerateTeamcityReport(report, reporterConfig.TeamcityReport)
			if err != nil {
				Fail(fmt.Sprintf("Failed to generate Teamcity report:\n%s", err.Error()))
			}
		}
	}

	flags := []string{}
	if reporterConfig.JSONReport != "" {
		flags = append(flags, "--json-report")
	}
	if reporterConfig.JUnitReport != "" {
		flags = append(flags, "--junit-report")
	}
	if reporterConfig.TeamcityReport != "" {
		flags = append(flags, "--teamcity-report")
	}
	pushNode(internal.NewNode(
		deprecationTracker, types.NodeTypeReportAfterSuite,
		fmt.Sprintf("Autogenerated ReportAfterSuite for %s", strings.Join(flags, " ")),
		body,
		types.NewCustomCodeLocation("autogenerated by Ginkgo"),
	))
}
